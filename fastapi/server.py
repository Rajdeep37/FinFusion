import os
import re
from typing import List, Optional, Dict
from decimal import Decimal
from enum import Enum
from dotenv import load_dotenv
import google.generativeai as genai
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configure the Gemini API
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env file")
genai.configure(api_key=GOOGLE_API_KEY)

# --- Constants ---
TOLERANCE = Decimal("0.01")  # Allow for minor rounding differences

# --- Enums ---
class MatchStatus(str, Enum):
    SUCCESS = "Success"
    DISCREPANCY_FOUND = "Discrepancy Found"
    PARTIAL_MATCH = "Partial Match"

class DiscrepancyType(str, Enum):
    QUANTITY_MISMATCH = "Quantity Mismatch"
    RATE_MISMATCH = "Rate Mismatch"
    ITEM_MISSING = "Item Missing"
    TOTAL_MISMATCH = "Total Mismatch"

# --- Pydantic Models ---

class LineItem(BaseModel):
    """Represents a single item in a document."""
    description: str = Field(..., min_length=1)
    stock_code: Optional[str] = None
    quantity: int = Field(..., gt=0)
    rate: Decimal = Field(..., gt=0)
    amount: Decimal = Field(..., gt=0)

    @validator('rate', 'amount', pre=True)
    def convert_to_decimal(cls, v):
        """Convert float/string to Decimal for precise calculations."""
        if isinstance(v, str):
            v = v.replace(',', '')
        return Decimal(str(v))

    class Config:
        json_encoders = {
            Decimal: lambda v: float(v)
        }

class ParsedDocument(BaseModel):
    """Base model for all parsed documents."""
    po_number: str = Field(..., pattern=r"^PO-\w+-\d+-\d+$")
    items: List[LineItem] = Field(..., min_items=1)
    total_amount: Decimal

    @validator('total_amount', pre=True)
    def convert_total_to_decimal(cls, v):
        if isinstance(v, str):
            v = v.replace(',', '')
        return Decimal(str(v))

    class Config:
        json_encoders = {
            Decimal: lambda v: float(v)
        }

class DocumentTexts(BaseModel):
    """Defines the structure of the incoming JSON request."""
    purchase_order_text: str = Field(..., min_length=10)
    goods_receipt_note_text: str = Field(..., min_length=10)
    purchase_invoice_text: str = Field(..., min_length=10)

class MatchDiscrepancy(BaseModel):
    """Describes a mismatch found during verification."""
    item_description: str
    discrepancy_type: DiscrepancyType
    issue: str
    details: Optional[Dict[str, any]] = None

class MatchResult(BaseModel):
    """Final output structure."""
    match_status: MatchStatus
    po_number: str
    discrepancies: List[MatchDiscrepancy]
    ai_summary: str
    statistics: dict = Field(default_factory=dict)
    
    class Config:
        arbitrary_types_allowed = True

# --- FastAPI Application ---
app = FastAPI(
    title="3-Way Match API",
    description="Processes text from PO, GRN, and Invoice documents to verify consistency.",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Document Parser ---

class DocumentParser:
    """Handles parsing of document text into structured data."""
    
    PO_NUMBER_PATTERNS = [
        r"PO-\w+-\d+-\d+",
        r"PO:\s*(PO-\w+-\d+-\d+)",
        r"Purchase Order:\s*(PO-\w+-\d+-\d+)"
    ]
    
    ITEM_PATTERN = re.compile(
        r"(.+?)\s+by\s+.+?\s+Stock Code:\s*(\S+)\s+(\d+)\s+₹?([\d,]+\.\d{2})\s+([\d,]+\.\d{2})",
        re.IGNORECASE
    )
    
    TOTAL_PATTERN = re.compile(r"TOTAL\s+₹?([\d,]+\.\d{2})", re.IGNORECASE)
    
    @classmethod
    def parse(cls, text: str, document_type: str = "Document") -> ParsedDocument:
        """
        Parses raw document text into structured data.
        
        Args:
            text: Raw document text
            document_type: Type of document for better error messages
            
        Returns:
            ParsedDocument instance
            
        Raises:
            HTTPException: If parsing fails
        """
        try:
            po_number = cls._extract_po_number(text)
            items = cls._extract_line_items(text)
            total = cls._extract_total(text)
            
            if not items:
                raise ValueError("No line items found in document")
            
            return ParsedDocument(
                po_number=po_number,
                items=items,
                total_amount=total
            )
            
        except ValueError as e:
            logger.error(f"Failed to parse {document_type}: {e}")
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Failed to parse {document_type}: {str(e)}"
            )
    
    @classmethod
    def _extract_po_number(cls, text: str) -> str:
        """Extract PO number using multiple patterns."""
        for pattern in cls.PO_NUMBER_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                po_number = match.group(1) if match.lastindex else match.group(0)
                return po_number.replace("PO: ", "").strip()
        raise ValueError("Purchase Order number not found")
    
    @classmethod
    def _extract_line_items(cls, text: str) -> List[LineItem]:
        """Extract all line items from document."""
        items = []
        for match in cls.ITEM_PATTERN.finditer(text):
            desc, stock, qty, rate, amt = match.groups()
            try:
                items.append(LineItem(
                    description=desc.strip(),
                    stock_code=stock.strip(),
                    quantity=int(qty),
                    rate=Decimal(rate.replace(",", "")),
                    amount=Decimal(amt.replace(",", ""))
                ))
            except (ValueError, TypeError) as e:
                logger.warning(f"Failed to parse line item: {e}")
                continue
        return items
    
    @classmethod
    def _extract_total(cls, text: str) -> Decimal:
        """Extract total amount from document."""
        match = cls.TOTAL_PATTERN.search(text.replace("\n", " "))
        if match:
            return Decimal(match.group(1).replace(",", ""))
        return Decimal("0.00")

# --- Three-Way Match Logic ---

class ThreeWayMatcher:
    """Performs three-way matching logic."""
    
    @staticmethod
    def validate_po_numbers(po: ParsedDocument, grn: ParsedDocument, 
                           invoice: ParsedDocument) -> None:
        """Ensure all documents have the same PO number."""
        if not (po.po_number == grn.po_number == invoice.po_number):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"PO Number Mismatch: PO='{po.po_number}', "
                       f"GRN='{grn.po_number}', Invoice='{invoice.po_number}'"
            )
    
    @staticmethod
    def match(po: ParsedDocument, grn: ParsedDocument, 
              invoice: ParsedDocument) -> List[MatchDiscrepancy]:
        """
        Performs the 3-way match and returns discrepancies.
        
        Match rules:
        - PO quantity >= GRN quantity = Invoice quantity
        - PO rate = Invoice rate
        - Invoice total <= PO total
        """
        ThreeWayMatcher.validate_po_numbers(po, grn, invoice)
        
        discrepancies = []
        
        # Create lookup dictionaries
        po_items = {item.stock_code: item for item in po.items}
        grn_items = {item.stock_code: item for item in grn.items}
        invoice_items = {item.stock_code: item for item in invoice.items}
        
        all_stock_codes = set(po_items.keys()) | set(grn_items.keys()) | set(invoice_items.keys())
        
        # Check each item
        for code in all_stock_codes:
            po_item = po_items.get(code)
            grn_item = grn_items.get(code)
            invoice_item = invoice_items.get(code)
            
            # Check if item exists in all documents
            if not all([po_item, grn_item, invoice_item]):
                missing_from = []
                if not po_item: missing_from.append("PO")
                if not grn_item: missing_from.append("GRN")
                if not invoice_item: missing_from.append("Invoice")
                
                discrepancies.append(MatchDiscrepancy(
                    item_description=f"Stock Code: {code}",
                    discrepancy_type=DiscrepancyType.ITEM_MISSING,
                    issue=f"Item missing from: {', '.join(missing_from)}",
                    details={"stock_code": code, "missing_from": missing_from}
                ))
                continue
            
            # Check quantities (PO >= GRN = Invoice)
            if not (po_item.quantity >= grn_item.quantity and 
                    grn_item.quantity == invoice_item.quantity):
                discrepancies.append(MatchDiscrepancy(
                    item_description=po_item.description,
                    discrepancy_type=DiscrepancyType.QUANTITY_MISMATCH,
                    issue=f"Quantity mismatch detected",
                    details={
                        "stock_code": code,
                        "po_quantity": po_item.quantity,
                        "grn_quantity": grn_item.quantity,
                        "invoice_quantity": invoice_item.quantity
                    }
                ))
            
            # Check rates (PO = Invoice)
            if abs(po_item.rate - invoice_item.rate) > TOLERANCE:
                discrepancies.append(MatchDiscrepancy(
                    item_description=po_item.description,
                    discrepancy_type=DiscrepancyType.RATE_MISMATCH,
                    issue=f"Rate mismatch detected",
                    details={
                        "stock_code": code,
                        "po_rate": float(po_item.rate),
                        "invoice_rate": float(invoice_item.rate),
                        "difference": float(abs(po_item.rate - invoice_item.rate))
                    }
                ))
        
        # Check total amounts
        if invoice.total_amount > po.total_amount + TOLERANCE:
            discrepancies.append(MatchDiscrepancy(
                item_description="Overall Total",
                discrepancy_type=DiscrepancyType.TOTAL_MISMATCH,
                issue="Invoice total exceeds PO total",
                details={
                    "po_total": float(po.total_amount),
                    "invoice_total": float(invoice.total_amount),
                    "difference": float(invoice.total_amount - po.total_amount)
                }
            ))
        
        return discrepancies

# --- AI Summary Generator ---

class AISummaryGenerator:
    """Handles AI summary generation using Gemini."""
    
    def __init__(self):
        self.model = genai.GenerativeModel('gemini-pro')
    
    async def generate(self, po_number: str, 
                      discrepancies: List[MatchDiscrepancy]) -> str:
        """
        Generate a human-readable summary of the match results.
        
        Args:
            po_number: Purchase order number
            discrepancies: List of found discrepancies
            
        Returns:
            AI-generated summary text
        """
        try:
            prompt = self._build_prompt(po_number, discrepancies)
            response = await self.model.generate_content_async(prompt)
            return response.text.strip()
        except Exception as e:
            logger.error(f"AI summary generation failed: {e}")
            return self._fallback_summary(po_number, discrepancies)
    
    def _build_prompt(self, po_number: str, 
                     discrepancies: List[MatchDiscrepancy]) -> str:
        """Build the prompt for Gemini."""
        if not discrepancies:
            return (
                f"The 3-way match for Purchase Order {po_number} was successful. "
                f"All items, quantities, rates, and amounts on the PO, GRN, and "
                f"vendor invoice are consistent. Write a brief, professional "
                f"one-sentence summary confirming this successful verification."
            )
        
        discrepancy_details = "\n".join([
            f"- {d.discrepancy_type.value}: {d.item_description} - {d.issue}"
            for d in discrepancies
        ])
        
        return (
            f"A 3-way match verification for Purchase Order {po_number} identified "
            f"the following discrepancies:\n\n{discrepancy_details}\n\n"
            f"Please generate a concise, professional summary (2-3 sentences) of "
            f"these findings. Start with the overall status and highlight the most "
            f"critical issues that require attention."
        )
    
    def _fallback_summary(self, po_number: str, 
                         discrepancies: List[MatchDiscrepancy]) -> str:
        """Generate a basic summary if AI fails."""
        if not discrepancies:
            return f"3-way match for PO {po_number} completed successfully with no discrepancies."
        
        return (
            f"3-way match for PO {po_number} found {len(discrepancies)} "
            f"discrepancy(ies) requiring attention. Please review the detailed "
            f"discrepancy list for resolution."
        )

# --- API Endpoints ---

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "3-Way Match API"}

@app.post("/process-documents/", response_model=MatchResult)
async def process_documents(texts: DocumentTexts):
    """
    Process PO, GRN, and Invoice texts to perform 3-way match verification.
    
    Args:
        texts: DocumentTexts containing raw text from all three documents
        
    Returns:
        MatchResult with status, discrepancies, and AI summary
    """
    logger.info("Starting 3-way match processing")
    
    try:
        # Parse documents
        parsed_po = DocumentParser.parse(texts.purchase_order_text, "Purchase Order")
        parsed_grn = DocumentParser.parse(texts.goods_receipt_note_text, "Goods Receipt Note")
        parsed_invoice = DocumentParser.parse(texts.purchase_invoice_text, "Purchase Invoice")
        
        logger.info(f"Successfully parsed all documents for PO {parsed_po.po_number}")
        
        # Perform three-way match
        matcher = ThreeWayMatcher()
        discrepancies = matcher.match(parsed_po, parsed_grn, parsed_invoice)
        
        # Generate AI summary
        summary_generator = AISummaryGenerator()
        summary = await summary_generator.generate(parsed_po.po_number, discrepancies)
        
        # Determine status
        if not discrepancies:
            match_status = MatchStatus.SUCCESS
        else:
            match_status = MatchStatus.DISCREPANCY_FOUND
        
        # Compile statistics
        statistics = {
            "total_items_checked": len(parsed_po.items),
            "discrepancies_found": len(discrepancies),
            "discrepancy_types": {
                dtype.value: sum(1 for d in discrepancies if d.discrepancy_type == dtype)
                for dtype in DiscrepancyType
            }
        }
        
        logger.info(f"Match completed: {match_status.value}, {len(discrepancies)} discrepancies")
        
        return MatchResult(
            match_status=match_status,
            po_number=parsed_po.po_number,
            discrepancies=discrepancies,
            ai_summary=summary,
            statistics=statistics
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during processing: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)