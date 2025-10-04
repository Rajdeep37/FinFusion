import { useState } from "react";
import * as pdfjsLib from "pdfjs-dist";
import pdfjsWorker from "pdfjs-dist/build/pdf.worker?url";
import "./App.css";

// Configure pdf.js worker
pdfjsLib.GlobalWorkerOptions.workerSrc = pdfjsWorker;

function App() {
  const [poText, setPoText] = useState("");
  const [grnText, setGrnText] = useState("");
  const [invoiceText, setInvoiceText] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [mode, setMode] = useState(""); // "2-way" or "3-way"

  // Extract text from uploaded PDF
  const extractTextFromPDF = async (file) => {
    const fileReader = new FileReader();
    return new Promise((resolve, reject) => {
      fileReader.onload = async function () {
        try {
          const typedArray = new Uint8Array(this.result);
          const pdf = await pdfjsLib.getDocument({ data: typedArray }).promise;

          let textContent = "";
          for (let i = 1; i <= pdf.numPages; i++) {
            const page = await pdf.getPage(i);
            const content = await page.getTextContent();
            textContent += content.items.map((s) => s.str).join(" ") + "\n";
          }
          resolve(textContent);
        } catch (err) {
          reject(err);
        }
      };
      fileReader.readAsArrayBuffer(file);
    });
  };

  // Handle file uploads
  const handleFileUpload = async (event, type) => {
    const file = event.target.files[0];
    if (!file) return;

    const text = await extractTextFromPDF(file);
    if (type === "po") setPoText(text);
    if (type === "grn") setGrnText(text);
    if (type === "invoice") setInvoiceText(text);
  };

  // Send to backend
  const sendToBackend = async (endpoint, payload, verificationType) => {
    setLoading(true);
    setResult(null);
    setMode(verificationType);
    try {
      const response = await fetch(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!response.ok) throw new Error("Backend error");
      const data = await response.json();
      setResult(data);
    } catch (err) {
      console.error("Error sending data to backend:", err);
      alert("Failed to process documents. Check console/logs.");
    } finally {
      setLoading(false);
    }
  };

  // 3-Way Verification
  const handleThreeWayVerification = async () => {
    if (!poText || !grnText || !invoiceText) {
      alert("Please upload PO, GRN, and Invoice for 3-way verification!");
      return;
    }

    const payload = {
      purchase_order_text: poText,
      goods_receipt_note_text: grnText,
      purchase_invoice_text: invoiceText,
    };

    await sendToBackend("http://localhost:8000/process-documents/", payload, "3-way");
  };

  // 2-Way Verification
  const handleTwoWayVerification = async () => {
    if (!poText || !invoiceText) {
      alert("Please upload PO and Invoice for 2-way verification!");
      return;
    }

    const payload = {
      purchase_order_text: poText,
      purchase_invoice_text: invoiceText,
    };

    await sendToBackend("http://localhost:8000/verify-procurement-text/", payload, "2-way");
  };

  return (
    <div className="App">
      <h1>Procurement Verification Portal</h1>

      <div className="upload-section">
        <label>Upload Purchase Order (PO): </label>
        <input type="file" accept="application/pdf" onChange={(e) => handleFileUpload(e, "po")} />
      </div>

      <div className="upload-section">
        <label>Upload Goods Receipt Note (GRN): </label>
        <input type="file" accept="application/pdf" onChange={(e) => handleFileUpload(e, "grn")} />
      </div>

      <div className="upload-section">
        <label>Upload Purchase Invoice: </label>
        <input type="file" accept="application/pdf" onChange={(e) => handleFileUpload(e, "invoice")} />
      </div>

      <div className="button-group">
        <button onClick={handleTwoWayVerification} disabled={loading}>
          {loading && mode === "2-way" ? "Processing..." : "2-Way Verification"}
        </button>
        <button onClick={handleThreeWayVerification} disabled={loading}>
          {loading && mode === "3-way" ? "Processing..." : "3-Way Verification"}
        </button>
      </div>

      {/* === RESULT DISPLAY === */}
      {result && (
        <div className="result-container">
          <h2>Verification Result</h2>

          {/* === 3-WAY RESULT === */}
          {mode === "3-way" && (
            <>
              <p><b>Status:</b> {result.match_status}</p>
              <p><b>PO Number:</b> {result.po_number}</p>
              <p><b>AI Summary:</b> {result.ai_summary}</p>

              <h3>Discrepancies</h3>
              {result.discrepancies?.length === 0 ? (
                <p>No discrepancies found ✅</p>
              ) : (
                <ul>
                  {result.discrepancies.map((d, idx) => (
                    <li key={idx}>
                      <b>{d.discrepancy_type}</b>: {d.item_description} - {d.issue}
                    </li>
                  ))}
                </ul>
              )}

              <h3>Statistics</h3>
              <pre>{JSON.stringify(result.statistics, null, 2)}</pre>
            </>
          )}

          {/* === 2-WAY RESULT === */}
          {mode === "2-way" && (
            <>
              <p><b>Verification Status:</b> {result.verification_status}</p>
              <p><b>Procurement Status:</b> {result.procurement_status}</p>
              <p><b>PO Number:</b> {result.po_number}</p>
              <p><b>AI Summary:</b> {result.ai_summary}</p>

              <h3>Items Checked</h3>
              <p><b>Total Items Checked:</b> {result.total_items_checked}</p>
              <p><b>Items with Discrepancies:</b> {result.items_with_discrepancies}</p>

              {result.quantity_discrepancies?.length > 0 && (
                <>
                  <h3>Quantity Discrepancies</h3>
                  <ul>
                    {result.quantity_discrepancies.map((item, idx) => (
                      <li key={idx}>
                        <b>{item.item_description}</b> (Code: {item.stock_code})<br />
                        PO Qty: {item.po_quantity}, Invoice Qty: {item.invoice_quantity}<br />
                        Variance: {item.variance} ({item.variance_percentage}%)
                      </li>
                    ))}
                  </ul>
                </>
              )}

              {result.excess_items?.length > 0 && (
                <>
                  <h3>Excess Items</h3>
                  <ul>
                    {result.excess_items.map((item, idx) => (
                      <li key={idx}>{item}</li>
                    ))}
                  </ul>
                </>
              )}

              {result.missing_items?.length > 0 && (
                <>
                  <h3>Missing Items</h3>
                  <ul>
                    {result.missing_items.map((item, idx) => (
                      <li key={idx}>{item}</li>
                    ))}
                  </ul>
                </>
              )}

              <h3>Financial Summary</h3>
              <pre>{JSON.stringify(result.financial_summary, null, 2)}</pre>

              {result.recommendations?.length > 0 && (
                <>
                  <h3>AI Recommendations</h3>
                  <ul>
                    {result.recommendations.map((rec, idx) => (
                      <li key={idx}>{rec}</li>
                    ))}
                  </ul>
                </>
              )}
            </>
          )}
        </div>
      )}
    </div>
  );
}

export default App;
