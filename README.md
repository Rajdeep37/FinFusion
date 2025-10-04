## 🧾 FinFusion

An intelligent **Procurement Document Verification Platform** that automates **2-Way** and **3-Way matching** between **Purchase Orders (PO)**, **Goods Receipt Notes (GRN)**, and **Purchase Invoices** using AI and document parsing.

### 🎯 Objective

To verify whether goods and invoices align with purchase orders — automatically detecting discrepancies in **quantities, prices, and items** using **PDF text extraction** and **AI-powered validation**.

---

## 🚀 Features

✅ **3-Way Verification** — Compares **PO**, **GRN**, and **Invoice** for full procurement validation
✅ **2-Way Verification** — Compares **PO** and **Invoice** when GRN is unavailable
✅ **AI-Powered Summaries** — Uses Gemini or other LLMs to generate concise, human-readable summaries
✅ **PDF Parsing** — Extracts structured text data using `pdfjs-dist`
✅ **Discrepancy Detection** — Highlights mismatches in items, rates, or totals
✅ **Simple Web Interface** — Upload and process documents directly in the browser

---

## 🏗️ Project Structure

```
procurement-verification/
│
├── backend/
│   ├── main.py                # FastAPI backend
│   ├── requirements.txt       # Python dependencies
│   └── ...                   
│
├── frontend/
│   ├── src/
│   │   ├── App.js             # React frontend (with 2 & 3-way verification)
│   │   └── App.css
│   ├── package.json
│   └── ...
│
└── README.md
```

---

## ⚙️ Tech Stack

### 🧠 Backend

* **FastAPI** — for RESTful API services
* **PyMuPDF / pdfplumber** — for PDF text extraction (if used server-side)
* **Gemini / OpenAI / LLM APIs** — for AI summary generation
* **Uvicorn** — for local API hosting

### 💻 Frontend

* **React.js** — for user interface
* **pdfjs-dist** — for client-side PDF text extraction
* **Fetch API** — for communication with backend

---

## 🧩 API Endpoints

### 🔹 `/process-documents/` (3-Way Match)

**Method:** `POST`
**Request Body:**

```json
{
  "purchase_order_text": "string",
  "goods_receipt_note_text": "string",
  "purchase_invoice_text": "string"
}
```

### 🔹 `/verify-procurement-text/` (2-Way Match)

**Method:** `POST`
**Request Body:**

```json
{
  "purchase_order_text": "string",
  "purchase_invoice_text": "string"
}
```

**Response Example:**

```json
{
  "match_status": "Partial Match",
  "po_number": "PO-ABC-123-2025",
  "ai_summary": "Invoice quantity for Widget B exceeds PO by 1 unit.",
  "discrepancies": [
    {
      "discrepancy_type": "Quantity Mismatch",
      "item_description": "Widget B",
      "issue": "PO: 5 vs Invoice: 6"
    }
  ],
  "statistics": {
    "total_items": 3,
    "matched_items": 2,
    "discrepant_items": 1
  }
}
```

---

## 🧰 Setup Instructions

### 🖥️ Backend (FastAPI)

```bash
cd fastapi
python -m venv venv
source venv/bin/activate   # (Windows: venv\Scripts\activate)
pip install -r requirements.txt
uvicorn main:app --reload
```

Your FastAPI backend will now run on:

```
http://localhost:8000
```

---

### 💻 Frontend (React)

```bash
cd frontend
npm install
npm run dev
```

The React app should run on:

```
http://localhost:5173
```

---

## 🧪 Usage Guide

1. Start the **FastAPI backend** (`uvicorn main:app --reload`)
2. Start the **React frontend** (`npm run dev`)
3. Open the app in your browser
4. Upload:

   * **PO**, **GRN**, and **Invoice** → Click **3-Way Verification**
   * **PO** and **Invoice only** → Click **2-Way Verification**
5. Wait for processing and view:

   * ✅ Match Status
   * 🧾 AI Summary
   * ⚠️ Discrepancies
   * 📊 Statistics

---

## 📊 Example Use Case

| Document    | Description                                  |
| ----------- | -------------------------------------------- |
| **PO**      | Contains ordered items and agreed quantities |
| **GRN**     | Lists received items and quantities          |
| **Invoice** | Lists billed items and prices                |

The system compares these to verify that:

* The **invoice matches** what was **ordered and received**
* Detects missing, extra, or mispriced items
* Provides a human-readable AI summary

---

## 🧠 Future Enhancements

* [ ] OCR Support for Scanned PDFs
* [ ] Multi-language document handling
* [ ] Authentication & user dashboard
* [ ] Excel/PDF report generation for audit trails
* [ ] Cloud deployment (AWS / Render / Vercel)

---

## 🏁 Author

**👨‍💻 Rajdeep Paul**
B.E. in **Artificial Intelligence & Data Science**
Passionate about **AI Automation**, **Data Engineering**, and **Intelligent Systems**

**👨‍💻 Nehil Chandrakar**
B.E. in **Artificial Intelligence & Machine Learning**
Passionate about **AI Automation**, **Data Engineering**, and **Intelligent Systems**
