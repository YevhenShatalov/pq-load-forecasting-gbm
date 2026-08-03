import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const buildDir = path.dirname(
  decodeURIComponent(new URL(import.meta.url).pathname).replace(/^\/(.:)/, "$1"),
);
const outputRoot = process.argv[2]
  ? path.resolve(process.argv[2])
  : path.resolve(buildDir, "..");
const tableDir = path.join(outputRoot, "tables");
const dataDir = path.join(outputRoot, "data");
const auditDir = path.join(outputRoot, "audit");
const previewDir = path.join(buildDir, "previews");
await fs.mkdir(previewDir, { recursive: true });

async function csvMatrix(filePath) {
  const csvText = await fs.readFile(filePath, "utf8");
  const temporary = await Workbook.fromCSV(csvText, { sheetName: "Imported" });
  const used = temporary.worksheets.getItem("Imported").getUsedRange(true);
  return used.values;
}

function columnName(number) {
  let value = number;
  let result = "";
  while (value > 0) {
    const remainder = (value - 1) % 26;
    result = String.fromCharCode(65 + remainder) + result;
    value = Math.floor((value - 1) / 26);
  }
  return result;
}

function styleDataSheet(sheet, matrix, title, sourceNote) {
  const rows = matrix.length;
  const columns = matrix[0]?.length || 1;
  const lastColumn = columnName(columns);
  sheet.showGridLines = false;
  sheet.getRange(`A1:${lastColumn}1`).merge();
  sheet.getRange("A1").values = [[title]];
  sheet.getRange(`A1:${lastColumn}1`).format = {
    fill: "#17365D",
    font: { name: "Arial", size: 15, bold: true, color: "#FFFFFF" },
    verticalAlignment: "center",
  };
  sheet.getRange(`A2:${lastColumn}2`).merge();
  sheet.getRange("A2").values = [[sourceNote]];
  sheet.getRange(`A2:${lastColumn}2`).format = {
    fill: "#DCE6F1",
    font: { name: "Arial", size: 9, italic: true, color: "#2F3E4D" },
    wrapText: true,
  };
  sheet.getRangeByIndexes(3, 0, rows, columns).values = matrix;
  sheet.getRange(`A4:${lastColumn}4`).format = {
    fill: "#4472C4",
    font: { name: "Arial", size: 9, bold: true, color: "#FFFFFF" },
    wrapText: true,
    verticalAlignment: "center",
    borders: { preset: "outside", style: "thin", color: "#A6B8CC" },
  };
  if (rows > 1) {
    sheet.getRange(`A5:${lastColumn}${rows + 3}`).format = {
      font: { name: "Arial", size: 9, color: "#1F1F1F" },
      verticalAlignment: "center",
      borders: {
        insideHorizontal: { style: "thin", color: "#E2E7EC" },
      },
    };
  }
  sheet.getRange(`A1:${lastColumn}${Math.min(rows + 3, 80)}`).format.autofitColumns();
  sheet.getRange(`A1:${lastColumn}${Math.min(rows + 3, 80)}`).format.autofitRows();
  sheet.getRange(`A:${lastColumn}`).format.columnWidth = 15;
  sheet.getRange(`A1:${lastColumn}2`).format.rowHeight = 26;
  sheet.getRange(`A4:${lastColumn}4`).format.rowHeight = 32;
  sheet.freezePanes.freezeRows(4);
  return { rows, columns, lastColumn };
}

function setWidths(sheet, widths) {
  for (const [column, width] of Object.entries(widths)) {
    sheet.getRange(`${column}:${column}`).format.columnWidth = width;
  }
}

const workbook = Workbook.create();
workbook.comments.setSelf({ displayName: "Yevhen Shatalov" });

const readme = workbook.worksheets.add("README");
readme.showGridLines = false;
readme.getRange("A1:H1").merge();
readme.getRange("A1").values = [["Conference Thesis Evidence Archive"]];
readme.getRange("A1:H1").format = {
  fill: "#17365D",
  font: { name: "Arial", size: 17, bold: true, color: "#FFFFFF" },
  verticalAlignment: "center",
};
readme.getRange("A3:B16").values = [
  ["Field", "Value"],
  ["Study", "Optuna search depth and chronological tuning-origin breadth for one fixed paired P-Q LightGBM architecture"],
  ["Primary period", "1 January-23 February 2022 (54 rolling origins)"],
  ["Descriptive stress period", "24-28 February 2022 (five rolling origins; excluded from ordinary inference)"],
  ["Configurations", "S24-T10, S24-T30, S24-T60, S36-T60, S48-T60, SNaive-24, SNaive-168"],
  ["P scale", "192.00558336970016 kW"],
  ["Q scale", "229.7979977161049 kVAr"],
  ["Point score", "Equal-weight mean of P and Q RMSE divided by the fixed target scales"],
  ["Probabilistic score", "Equal-weight mean of P and Q CRPS divided by the fixed target scales"],
  ["Daily tests", "One complete 24-h trajectory per observation; Bartlett HAC lag 3; HLN correction; two-sided p-values"],
  ["Multiple testing", "Separate three-comparison Holm families for depth/breadth and point/CRPS losses"],
  ["Leakage audit", "PASS: future-target perturbation and P-Q target-order invariance differences both equal zero"],
  ["Tuning status", "No new hyperparameter tuning performed during finalization"],
  ["Generated", `UTC ${new Date().toISOString()}`],
];
readme.getRange("A3:B3").format = {
  fill: "#4472C4",
  font: { name: "Arial", size: 10, bold: true, color: "#FFFFFF" },
};
readme.getRange("A4:A16").format = {
  fill: "#DCE6F1",
  font: { name: "Arial", size: 9, bold: true, color: "#17365D" },
};
readme.getRange("B4:B16").format = {
  font: { name: "Arial", size: 9 },
  wrapText: true,
};
readme.getRange("A3:B16").format.borders = {
  insideHorizontal: { style: "thin", color: "#D9E1EA" },
  outside: { style: "thin", color: "#9FB3C8" },
};
setWidths(readme, { A: 26, B: 92 });
readme.getRange("A1:H1").format.rowHeight = 32;
readme.getRange("A3:B16").format.autofitRows();
readme.freezePanes.freezeRows(3);

const sources = [
  { name: "Table1 Design", path: path.join(tableDir, "Table1_search_design_cost.csv"), title: "Table 1 - Search Design and Computational Cost", note: "Machine-derived from the nested seed-42 anytime curve and breadth search-completion records." },
  { name: "Table2 Performance", path: path.join(tableDir, "Table2_primary_performance.csv"), title: "Table 2 - Primary 54-Day Performance", note: "Lower values are better. P is in kW; Q is in kVAr; paired scores use fixed development scales." },
  { name: "Table3 Tests", path: path.join(tableDir, "Table3_paired_tests.csv"), title: "Table 3 - Daily Paired Comparisons", note: "Separate Holm families are used for each design question and loss definition." },
  { name: "Breadth Seeds", path: path.join(tableDir, "breadth_three_seed_summary.csv"), title: "Three-Seed Development Breadth Summary", note: "Descriptive development evidence only; seed trajectories are not independent forecasting observations." },
  { name: "Primary Metrics", path: path.join(dataDir, "primary_54day_metrics.csv"), title: "Primary Metrics - 54 Origins", note: "1 January-23 February 2022. This sheet is the source for the main numerical ranking." },
  { name: "Stress Metrics", path: path.join(dataDir, "stress_5day_metrics.csv"), title: "Descriptive Stress Metrics - Five Origins", note: "24-28 February 2022. These values are descriptive and are excluded from ordinary inference." },
  { name: "Daily Losses", path: path.join(dataDir, "daily_paired_losses.csv"), title: "Daily Paired Point and CRPS Losses", note: "One row per configuration and midnight origin; analysis_period separates the primary and descriptive intervals." },
  { name: "Forecast Wide", path: path.join(dataDir, "forecast_59days_wide.csv"), title: "Complete 59-Day Forecasts - Wide Form", note: "One row per hourly timestamp with prepared observations and every configuration prediction." },
  { name: "Forecast Long", path: path.join(dataDir, "forecast_59days_long.csv"), title: "Complete 59-Day Forecasts - Long Form", note: "One row per configuration, target, and forecast timestamp; 1,416 hours per target and configuration." },
  { name: "Source Inventory", path: path.join(auditDir, "source_inventory.csv"), title: "Source Inventory and Hashes", note: "Frozen files used to reconstruct the publication package." },
  { name: "Origin Audit", path: path.join(auditDir, "forecast_origin_audit.csv"), title: "Forecast-Origin Structural Audit", note: "Cutoff, horizon, row-count, and target-alignment checks for all five learned configurations." },
];

const renderedSheets = ["README"];
const previewRanges = { README: "A1:H16" };
for (const item of sources) {
  const matrix = await csvMatrix(item.path);
  const sheet = workbook.worksheets.add(item.name);
  const geometry = styleDataSheet(sheet, matrix, item.title, item.note);
  renderedSheets.push(item.name);
  previewRanges[item.name] = `A1:${geometry.lastColumn}${Math.min(geometry.rows + 3, 24)}`;
  if (item.name === "Table1 Design") {
    setWidths(sheet, { A: 16, B: 14, C: 14, D: 50, E: 18, F: 38, G: 26 });
    sheet.getRange(`B5:C${geometry.rows + 3}`).format.numberFormat = "0";
    sheet.getRange(`E5:E${geometry.rows + 3}`).format.numberFormat = "#,##0";
    sheet.getRange(`G5:G${geometry.rows + 3}`).format.numberFormat = "#,##0.0";
  } else if (item.name === "Table2 Performance") {
    setWidths(sheet, { A: 17, B: 15, C: 16, D: 20, E: 14, F: 14, G: 21 });
    sheet.getRange(`B5:C${geometry.rows + 3}`).format.numberFormat = "0.00";
    sheet.getRange(`D5:D${geometry.rows + 3}`).format.numberFormat = "0.0000";
    sheet.getRange(`E5:F${geometry.rows + 3}`).format.numberFormat = "0.00";
    sheet.getRange(`G5:G${geometry.rows + 3}`).format.numberFormat = "0.0000";
  } else if (item.name === "Table3 Tests") {
    setWidths(sheet, { A: 20, B: 31, C: 14, D: 14, E: 26, F: 20, G: 14, H: 14, I: 13, J: 10, K: 11, L: 12, M: 17, N: 13, O: 15 });
    sheet.getRange(`F5:I${geometry.rows + 3}`).format.numberFormat = "0.000000";
    sheet.getRange(`N5:N${geometry.rows + 3}`).format.numberFormat = "0.000000";
  } else if (item.name === "Breadth Seeds") {
    setWidths(sheet, { A: 10, B: 18, C: 28, D: 20, E: 23, F: 27, G: 22, H: 24, I: 31 });
    sheet.getRange(`D5:I${geometry.rows + 3}`).format.numberFormat = "0.000000";
  } else if (item.name.includes("Metrics")) {
    setWidths(sheet, { A: 17, B: 24, C: 12, D: 18, E: 15, F: 14, G: 16, H: 14, I: 14, J: 14, K: 16 });
  } else if (item.name === "Daily Losses") {
    setWidths(sheet, { A: 17, B: 13, C: 14, D: 14, E: 14, F: 14, G: 20, H: 14, I: 14, J: 20, K: 23 });
  } else if (item.name === "Forecast Wide") {
    setWidths(sheet, { A: 15, B: 19, C: 10, D: 23, E: 14, F: 15 });
  } else if (item.name === "Forecast Long") {
    setWidths(sheet, { A: 17, B: 23, C: 15, D: 19, E: 10, F: 14, G: 14, H: 14, I: 14, J: 19, K: 19, L: 42 });
  } else if (item.name === "Source Inventory") {
    setWidths(sheet, { A: 30, B: 68, C: 68, D: 10, E: 25, F: 18, G: 19, H: 10, I: 22, J: 22, K: 35 });
  }
}

const auditJson = JSON.parse(await fs.readFile(path.join(auditDir, "forecast_leakage_audit.json"), "utf8"));
const auditSheet = workbook.worksheets.add("Leakage Audit");
const auditRows = Object.entries(auditJson).map(([key, value]) => [
  key,
  key === "created_at"
    ? `UTC ${value}`
    : Array.isArray(value)
      ? value.join(", ")
      : typeof value === "object" && value !== null
        ? JSON.stringify(value)
        : value,
]);
styleDataSheet(
  auditSheet,
  [["check", "value"], ...auditRows],
  "Leakage and Recursive-Order Audit",
  "PASS is required before manuscript generation. Behavioral checks refit frozen specifications but perform no tuning.",
);
setWidths(auditSheet, { A: 44, B: 72 });
renderedSheets.push("Leakage Audit");
previewRanges["Leakage Audit"] = "A1:B24";

const outputPath = path.join(tableDir, "full_results.xlsx");
const exported = await SpreadsheetFile.exportXlsx(workbook);
await exported.save(outputPath);

for (const sheetName of renderedSheets) {
  const preview = await workbook.render({
    sheetName,
    range: previewRanges[sheetName],
    scale: 1,
    format: "png",
  });
  const safeName = sheetName.replace(/[^A-Za-z0-9]+/g, "_");
  await fs.writeFile(path.join(previewDir, `${safeName}.png`), new Uint8Array(await preview.arrayBuffer()));
}

const inspection = await workbook.inspect({
  kind: "sheet,table",
  maxChars: 5000,
  tableMaxRows: 5,
  tableMaxCols: 8,
});
await fs.writeFile(path.join(buildDir, "workbook_inspection.ndjson"), inspection.ndjson, "utf8");
const errors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 100 },
  summary: "final formula error scan",
});
await fs.writeFile(path.join(buildDir, "formula_error_scan.ndjson"), errors.ndjson, "utf8");

console.log(outputPath);
