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

  // Send extracted text to backend
  const sendToBackend = async () => {
    if (!poText || !grnText || !invoiceText) {
      alert("Please upload all three documents first!");
      return;
    }

    setLoading(true);
    try {
      const response = await fetch("http://localhost:8000/process-documents/", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          purchase_order_text: poText,
          goods_receipt_note_text: grnText,
          purchase_invoice_text: invoiceText,
        }),
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

  return (
    <div className="App">
      <h1>3-Way Match Verifier</h1>

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

      <button onClick={sendToBackend} disabled={loading}>
        {loading ? "Processing..." : "Send to Backend"}
      </button>

      {result && (
        <div className="result-container">
          <h2>Result</h2>
          <p><b>Status:</b> {result.match_status}</p>
          <p><b>PO Number:</b> {result.po_number}</p>
          <p><b>AI Summary:</b> {result.ai_summary}</p>

          <h3>Discrepancies</h3>
          {result.discrepancies.length === 0 ? (
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
        </div>
      )}
    </div>
  );
}

export default App;
