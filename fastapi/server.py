import os
import re
from typing import List, Optional, Dict, Any
from decimal import Decimal
from enum import Enum
from dotenv import load_dotenv
import google.generativeai as genai
from fastapi import FastAPI, HTTPException, status, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
import logging
import PyPDF2
from io import BytesIO
from pydantic import BaseModel
from typing import List

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

class ProcurementStatus(str, Enum):
    EXACT_MATCH = "Exact Match"
    SHORT_PROCUREMENT = "Short Procurement"
    EXCESS_PROCUREMENT = "Excess Procurement"
    MIXED = "Mixed (Some Excess, Some Short)"


class InventoryAnalysisResult(BaseModel):
    stock_code: str
    description: str
    quantity: int
    rate: float
    total_amount: float
    carrying_cost: float
    gross_margin: float
    alert: bool  # True if gross margin < carrying cost

class InvoiceRegisterRequest(BaseModel):
    invoice_text: str  # Raw text from PDF

class InvoiceRegisterResponse(BaseModel):
    total_items: int
    total_invoice_amount: float
    total_carrying_cost: float
    total_gross_margin: float
    analysis: List[InventoryAnalysisResult]
    
# --- Pydantic Models ---

class LineItem(BaseModel):
    """Represents a single item in a document."""
    description: str = Field(..., min_length=1)
    stock_code: Optional[str] = None
    quantity: int = Field(..., gt=0)
    rate: Decimal = Field(..., gt=0)
    amount: Decimal = Field(..., gt=0)

    @field_validator('rate', 'amount')
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

    @field_validator("total_amount")
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
    details: Optional[Dict[str, Any]] = None

class MatchResult(BaseModel):
    """Final output structure."""
    match_status: MatchStatus
    po_number: str
    discrepancies: List[MatchDiscrepancy]
    ai_summary: str
    statistics: dict = Field(default_factory=dict)
    
    class Config:
        arbitrary_types_allowed = True

# --- NEW MODELS FOR 2-WAY VERIFICATION ---

class ProcurementDiscrepancy(BaseModel):
    """Describes procurement quantity/amount discrepancies."""
    item_description: str
    stock_code: Optional[str]
    po_quantity: int
    invoice_quantity: int
    variance: int  # Positive = excess, Negative = short
    variance_percentage: float
    po_amount: float
    invoice_amount: float
    amount_variance: float

class ProcurementTextRequest(BaseModel):
    purchase_order_text: str
    purchase_invoice_text: str

class TwoWayVerificationResult(BaseModel):
    """Result of 2-way PO vs Invoice verification."""
    verification_status: str
    procurement_status: ProcurementStatus
    po_number: str
    total_items_checked: int
    items_with_discrepancies: int
    quantity_discrepancies: List[ProcurementDiscrepancy]
    rate_discrepancies: List[ProcurementDiscrepancy]
    missing_items: List[str]
    excess_items: List[str]
    financial_summary: dict
    ai_summary: str
    recommendations: List[str]

# --- FastAPI Application ---
app = FastAPI(
    title="Procurement Verification API",
    description="Processes PO, GRN, and Invoice documents for 3-way and 2-way verification.",
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

# --- PDF Parser ---

class PDFParser:
    """Handles PDF file parsing to extract text."""
    
    @staticmethod
    async def extract_text_from_pdf(file: UploadFile) -> str:
        """
        Extract text from uploaded PDF file.
        
        Args:
            file: UploadFile object from FastAPI
            
        Returns:
            Extracted text as string
            
        Raises:
            HTTPException: If PDF parsing fails
        """
        try:
            # Read file content
            content = await file.read()
            pdf_file = BytesIO(content)
            
            # Parse PDF
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            if len(pdf_reader.pages) == 0:
                raise ValueError("PDF file is empty")
            
            # Extract text from all pages
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            
            if not text.strip():
                raise ValueError("No text could be extracted from PDF")
            
            logger.info(f"Successfully extracted {len(text)} characters from PDF")
            return text
            
        except Exception as e:
            logger.error(f"Failed to parse PDF: {e}")
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Failed to parse PDF file: {str(e)}"
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
    r"\d*\.\s*(.+?)\s+by\s+.+?\s+Stock Code:\s*(\S+)\s+(\d+)\s+₹?([\d,]+\.\d{2})\s+₹?([\d,]+\.\d{2})",
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

# --- NEW: Two-Way Verification Logic ---

class TwoWayVerifier:
    """Performs 2-way verification between PO and Invoice."""
    
    @staticmethod
    def verify(po: ParsedDocument, invoice: ParsedDocument) -> TwoWayVerificationResult:
        """
        Verifies PO against Invoice and identifies procurement discrepancies.
        
        Args:
            po: Parsed Purchase Order
            invoice: Parsed Purchase Invoice
            
        Returns:
            TwoWayVerificationResult with detailed analysis
        """
        # Validate PO numbers match
        if po.po_number != invoice.po_number:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"PO Number Mismatch: PO='{po.po_number}', Invoice='{invoice.po_number}'"
            )
        
        # Create lookup dictionaries
        po_items = {item.stock_code: item for item in po.items}
        invoice_items = {item.stock_code: item for item in invoice.items}
        
        # Initialize result lists
        quantity_discrepancies = []
        rate_discrepancies = []
        missing_items = []
        excess_items = []
        
        # Track overall procurement status
        has_excess = False
        has_short = False
        
        # Check items in PO
        for code, po_item in po_items.items():
            if code not in invoice_items:
                missing_items.append(f"{po_item.description} (Code: {code})")
                has_short = True
                continue
            
            invoice_item = invoice_items[code]
            
            # Check quantity variance
            variance = invoice_item.quantity - po_item.quantity
            if variance != 0:
                variance_pct = (variance / po_item.quantity) * 100
                
                if variance > 0:
                    has_excess = True
                else:
                    has_short = True
                
                quantity_discrepancies.append(ProcurementDiscrepancy(
                    item_description=po_item.description,
                    stock_code=code,
                    po_quantity=po_item.quantity,
                    invoice_quantity=invoice_item.quantity,
                    variance=variance,
                    variance_percentage=round(variance_pct, 2),
                    po_amount=float(po_item.amount),
                    invoice_amount=float(invoice_item.amount),
                    amount_variance=float(invoice_item.amount - po_item.amount)
                ))
            
            # Check rate variance
            if abs(po_item.rate - invoice_item.rate) > TOLERANCE:
                rate_discrepancies.append(ProcurementDiscrepancy(
                    item_description=po_item.description,
                    stock_code=code,
                    po_quantity=po_item.quantity,
                    invoice_quantity=invoice_item.quantity,
                    variance=0,
                    variance_percentage=0.0,
                    po_amount=float(po_item.rate),
                    invoice_amount=float(invoice_item.rate),
                    amount_variance=float(invoice_item.rate - po_item.rate)
                ))
        
        # Check for excess items in invoice not in PO
        for code, invoice_item in invoice_items.items():
            if code not in po_items:
                excess_items.append(f"{invoice_item.description} (Code: {code})")
                has_excess = True
        
        # Determine procurement status
        if has_excess and has_short:
            procurement_status = ProcurementStatus.MIXED
        elif has_excess:
            procurement_status = ProcurementStatus.EXCESS_PROCUREMENT
        elif has_short:
            procurement_status = ProcurementStatus.SHORT_PROCUREMENT
        else:
            procurement_status = ProcurementStatus.EXACT_MATCH
        
        # Calculate financial summary
        total_po_amount = float(po.total_amount)
        total_invoice_amount = float(invoice.total_amount)
        amount_variance = total_invoice_amount - total_po_amount
        variance_percentage = (amount_variance / total_po_amount * 100) if total_po_amount > 0 else 0
        
        financial_summary = {
            "po_total": total_po_amount,
            "invoice_total": total_invoice_amount,
            "amount_variance": amount_variance,
            "variance_percentage": round(variance_percentage, 2),
            "within_budget": amount_variance <= 0
        }
        
        # Generate recommendations
        recommendations = TwoWayVerifier._generate_recommendations(
            procurement_status,
            quantity_discrepancies,
            rate_discrepancies,
            missing_items,
            excess_items,
            financial_summary
        )
        
        # Determine overall verification status
        total_issues = len(quantity_discrepancies) + len(rate_discrepancies) + len(missing_items) + len(excess_items)
        verification_status = "Approved" if total_issues == 0 else "Requires Review"
        
        return TwoWayVerificationResult(
            verification_status=verification_status,
            procurement_status=procurement_status,
            po_number=po.po_number,
            total_items_checked=len(po_items),
            items_with_discrepancies=len(quantity_discrepancies) + len(rate_discrepancies),
            quantity_discrepancies=quantity_discrepancies,
            rate_discrepancies=rate_discrepancies,
            missing_items=missing_items,
            excess_items=excess_items,
            financial_summary=financial_summary,
            ai_summary="",  # Will be filled by AI
            recommendations=recommendations
        )
    
    @staticmethod
    def _generate_recommendations(
        procurement_status: ProcurementStatus,
        quantity_discrepancies: List[ProcurementDiscrepancy],
        rate_discrepancies: List[ProcurementDiscrepancy],
        missing_items: List[str],
        excess_items: List[str],
        financial_summary: dict
    ) -> List[str]:
        """Generate actionable recommendations based on findings."""
        recommendations = []
        
        if procurement_status == ProcurementStatus.EXACT_MATCH:
            recommendations.append("✅ All quantities match. Proceed with payment approval.")
        
        if procurement_status == ProcurementStatus.EXCESS_PROCUREMENT:
            recommendations.append("⚠️ Excess procurement detected. Verify authorization for additional items.")
            recommendations.append("📋 Review excess items with procurement team before payment.")
        
        if procurement_status == ProcurementStatus.SHORT_PROCUREMENT:
            recommendations.append("⚠️ Short delivery detected. Verify if partial delivery was authorized.")
            recommendations.append("📞 Contact vendor regarding missing items.")
            recommendations.append("💰 Adjust payment amount to reflect actual delivered quantity.")
        
        if procurement_status == ProcurementStatus.MIXED:
            recommendations.append("🔍 Mixed procurement issues detected. Detailed review required.")
            recommendations.append("📊 Reconcile each discrepancy with procurement and receiving teams.")
        
        if rate_discrepancies:
            recommendations.append(f"💲 {len(rate_discrepancies)} rate mismatch(es) found. Verify with vendor before payment.")
        
        if not financial_summary.get("within_budget"):
            variance = financial_summary.get("amount_variance", 0)
            recommendations.append(f"⚠️ Invoice exceeds PO by ₹{abs(variance):,.2f}. Budget approval required.")
        
        if missing_items:
            recommendations.append(f"📦 {len(missing_items)} item(s) from PO not invoiced. Follow up with vendor.")
        
        if excess_items:
            recommendations.append(f"➕ {len(excess_items)} extra item(s) in invoice. Verify if these should be billed.")
        
        return recommendations if recommendations else ["✅ No issues found. Approve for payment."]

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
    
    async def generate_two_way_summary(self, result: TwoWayVerificationResult) -> str:
        """
        Generate AI summary for 2-way verification.
        
        Args:
            result: TwoWayVerificationResult object
            
        Returns:
            AI-generated summary text
        """
        try:
            prompt = self._build_two_way_prompt(result)
            response = await self.model.generate_content_async(prompt)
            return response.text.strip()
        except Exception as e:
            logger.error(f"AI summary generation failed: {e}")
            return self._fallback_two_way_summary(result)
    
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
    
    def _build_two_way_prompt(self, result: TwoWayVerificationResult) -> str:
        """Build prompt for 2-way verification summary."""
        issues = []
        
        if result.quantity_discrepancies:
            for disc in result.quantity_discrepancies:
                status = "excess" if disc.variance > 0 else "short"
                issues.append(
                    f"- {disc.item_description}: {status} by {abs(disc.variance)} units "
                    f"({abs(disc.variance_percentage):.1f}%)"
                )
        
        if result.rate_discrepancies:
            for disc in result.rate_discrepancies:
                issues.append(
                    f"- {disc.item_description}: rate difference of ₹{abs(disc.amount_variance):.2f}"
                )
        
        if result.missing_items:
            issues.append(f"- Missing items from invoice: {', '.join(result.missing_items[:3])}")
        
        if result.excess_items:
            issues.append(f"- Extra items in invoice: {', '.join(result.excess_items[:3])}")
        
        variance_info = (
            f"Invoice amount is ₹{abs(result.financial_summary['amount_variance']):,.2f} "
            f"{'higher' if result.financial_summary['amount_variance'] > 0 else 'lower'} than PO "
            f"({abs(result.financial_summary['variance_percentage']):.1f}% variance)."
        )
        
        if issues:
            issues_text = "\n".join(issues)
            prompt = (
                f"A 2-way verification (PO vs Invoice) for Purchase Order {result.po_number} "
                f"found the following:\n\n"
                f"Procurement Status: {result.procurement_status.value}\n"
                f"{variance_info}\n\n"
                f"Issues:\n{issues_text}\n\n"
                f"Generate a concise, professional 2-3 sentence summary highlighting the "
                f"procurement status and key financial implications. Focus on actionable insights."
            )
        else:
            prompt = (
                f"A 2-way verification (PO vs Invoice) for Purchase Order {result.po_number} "
                f"shows perfect alignment. All items, quantities, and rates match exactly. "
                f"Generate a brief 1-sentence confirmation suitable for approval."
            )
        
        return prompt
    
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
    
    def _fallback_two_way_summary(self, result: TwoWayVerificationResult) -> str:
        """Generate basic summary for 2-way verification if AI fails."""
        if result.verification_status == "Approved":
            return f"2-way verification for PO {result.po_number} approved. All items and amounts match."
        
        return (
            f"2-way verification for PO {result.po_number} requires review. "
            f"Procurement status: {result.procurement_status.value}. "
            f"{result.items_with_discrepancies} item(s) have discrepancies."
        )

# --- API Endpoints ---

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "Procurement Verification API"}

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

# --- NEW ENDPOINT: 2-Way Verification with PDF Upload ---

@app.post("/verify-procurement/", response_model=TwoWayVerificationResult)
async def verify_procurement(
    po_pdf: UploadFile = File(..., description="Purchase Order PDF file"),
    invoice_pdf: UploadFile = File(..., description="Purchase Invoice PDF file")
):
    """
    Perform 2-way verification between PO and Invoice PDFs.
    
    Identifies:
    - Excess procurement (invoice quantity > PO quantity)
    - Short procurement (invoice quantity < PO quantity)
    - Rate mismatches
    - Missing or extra items
    
    Args:
        po_pdf: Purchase Order PDF file
        invoice_pdf: Purchase Invoice PDF file
        
    Returns:
        TwoWayVerificationResult with detailed procurement analysis
    """
    logger.info("Starting 2-way procurement verification")
    
    try:
        # Validate file types
        if not po_pdf.filename.lower().endswith('.pdf'):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Purchase Order must be a PDF file"
            )
        
        if not invoice_pdf.filename.lower().endswith('.pdf'):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Purchase Invoice must be a PDF file"
            )
        
        # Extract text from PDFs
        logger.info(f"Extracting text from PO: {po_pdf.filename}")
        po_text = await PDFParser.extract_text_from_pdf(po_pdf)
        
        logger.info(f"Extracting text from Invoice: {invoice_pdf.filename}")
        invoice_text = await PDFParser.extract_text_from_pdf(invoice_pdf)
        
        # Parse documents
        parsed_po = DocumentParser.parse(po_text, "Purchase Order")
        parsed_invoice = DocumentParser.parse(invoice_text, "Purchase Invoice")
        
        logger.info(f"Successfully parsed both documents for PO {parsed_po.po_number}")
        
        # Perform 2-way verification
        verifier = TwoWayVerifier()
        result = verifier.verify(parsed_po, parsed_invoice)
        
        # Generate AI summary
        summary_generator = AISummaryGenerator()
        ai_summary = await summary_generator.generate_two_way_summary(result)
        
        # Update result with AI summary
        result.ai_summary = ai_summary
        
        logger.info(
            f"Verification completed: {result.verification_status}, "
            f"Procurement Status: {result.procurement_status.value}"
        )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during procurement verification: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {str(e)}"
        )

# --- ADDITIONAL ENDPOINT: 2-Way Verification with Text Input ---

@app.post("/verify-procurement-text/", response_model=TwoWayVerificationResult)
async def verify_procurement_text(request: ProcurementTextRequest):
    """
    Perform 2-way verification between PO and Invoice using text input.

    Args:
        request: JSON payload containing raw text from Purchase Order & Invoice

    Returns:
        TwoWayVerificationResult with detailed procurement analysis
    """
    logger.info("Starting 2-way procurement verification (text input via JSON)")

    try:
        # Parse documents
        print("Parsing documents from text input")
        parsed_po = DocumentParser.parse(request.purchase_order_text, "Purchase Order")
        print("Parsed PO successfully")
        parsed_invoice = DocumentParser.parse(request.purchase_invoice_text, "Purchase Invoice")
        print("Parsed Invoice successfully")

        logger.info(f"Successfully parsed both documents for PO {parsed_po.po_number}")

        # Perform 2-way verification
        print("Performing 2-way verification")
        verifier = TwoWayVerifier()
        print("Verifier initialized")
        result = verifier.verify(parsed_po, parsed_invoice)
        print("Verification completed")

        # Generate AI summary
        summary_generator = AISummaryGenerator()
        ai_summary = await summary_generator.generate_two_way_summary(result)

        # Update result with AI summary
        result.ai_summary = ai_summary

        logger.info(
            f"Verification completed: {result.verification_status}, "
            f"Procurement Status: {result.procurement_status.value}"
        )

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during procurement verification: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {str(e)}"
        )
    


@app.post("/invoice-register-text/", response_model=InvoiceRegisterResponse)
async def invoice_register_text(request: InvoiceRegisterRequest):
    """
    Parses invoice text, computes carrying costs and gross margins,
    and highlights items where Gross Margin < Carrying Cost.
    """
    try:
        invoice_text = request.invoice_text
        if not invoice_text.strip():
            raise HTTPException(status_code=422, detail="Invoice text cannot be empty")

        # Parse line items
        items = DocumentParser._extract_line_items(invoice_text)
        if not items:
            raise HTTPException(status_code=422, detail="No line items found in invoice")

        analysis_results = []
        total_invoice_amount = Decimal("0.00")
        total_carrying_cost = Decimal("0.00")
        total_gross_margin = Decimal("0.00")

        for item in items:
            # Example: carrying cost = 10% of rate per unit
            carrying_cost_per_unit = item.rate * Decimal("0.1")
            carrying_cost_total = carrying_cost_per_unit * item.quantity

            # Example: cost price = 80% of rate
            cost_price = item.rate * Decimal("0.8")
            gross_margin_total = (item.rate - cost_price) * item.quantity

            alert = gross_margin_total < carrying_cost_total

            analysis_results.append(InventoryAnalysisResult(
                stock_code=item.stock_code,
                description=item.description,
                quantity=item.quantity,
                rate=float(item.rate),
                total_amount=float(item.amount),
                carrying_cost=float(carrying_cost_total),
                gross_margin=float(gross_margin_total),
                alert=alert
            ))

            total_invoice_amount += item.amount
            total_carrying_cost += carrying_cost_total
            total_gross_margin += gross_margin_total

        return InvoiceRegisterResponse(
            total_items=len(items),
            total_invoice_amount=float(total_invoice_amount),
            total_carrying_cost=float(total_carrying_cost),
            total_gross_margin=float(total_gross_margin),
            analysis=analysis_results
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Invoice register analysis failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {str(e)}"
        )




if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)