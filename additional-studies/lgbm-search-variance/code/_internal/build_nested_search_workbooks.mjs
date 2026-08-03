import path from "node:path";
import { pathToFileURL } from "node:url";


function requireArgument(value, name) {
  if (!value) {
    throw new Error(`Missing required argument: ${name}`);
  }
  return value;
}


function asExcelSerial(value) {
  const [datePart, timePart = "00:00:00"] = value.split("T");
  const [year, month, day] = datePart.split("-").map(Number);
  const [hour, minute, second] = timePart.split(":").map(Number);
  return (
    Date.UTC(year, month - 1, day, hour, minute, second) / 86400000 +
    25569
  );
}


function asExcelValue(column, value) {
  if (value === null || value === undefined) {
    return null;
  }
  if (["train_start", "train_end", "test_start", "test_end"].includes(column)) {
    return asExcelSerial(value);
  }
  if (column === "origin_id") {
    return asExcelSerial(`${value}T00:00:00`);
  }
  return value;
}


async function buildWorkbook({
  sourcePath,
  outputPath,
  artifactToolEntry,
  columns,
  rows,
  tableName,
}) {
  const artifactTool = await import(pathToFileURL(artifactToolEntry).href);
  const input = await artifactTool.FileBlob.load(sourcePath);
  const workbook = await artifactTool.SpreadsheetFile.importXlsx(input);
  const sheet = workbook.worksheets.getItem("24");

  sheet.tables.deleteAll();
  const matrix = [
    columns,
    ...rows.map((row) =>
      columns.map((column) => asExcelValue(column, row[column])),
    ),
  ];
  const lastRow = matrix.length;
  const dataRange = sheet.getRange(`A1:O${lastRow}`);
  dataRange.values = matrix;

  sheet.getRange(`A2:D${lastRow}`).format.numberFormat =
    "yyyy-mm-dd hh:mm";
  sheet.getRange(`E2:E${lastRow}`).format.numberFormat = "yyyy-mm-dd";
  sheet.getRange(`A1:O${lastRow}`).format.verticalAlignment = "center";
  sheet.getRange(`A1:O${lastRow}`).format.wrapText = false;
  sheet.getRange(`J2:J${lastRow}`).format.wrapText = true;

  const widths = {
    A: 20,
    B: 20,
    C: 20,
    D: 20,
    E: 13,
    F: 14,
    G: 22,
    H: 12,
    I: 30,
    J: 62,
    K: 14,
    L: 22,
    M: 14,
    N: 12,
    O: 16,
  };
  for (const [column, width] of Object.entries(widths)) {
    sheet.getRange(`${column}:${column}`).format.columnWidth = width;
  }
  sheet.getRange(`1:${lastRow}`).format.rowHeight = 18;
  sheet.getRange(`2:${lastRow}`).format.rowHeight = 30;

  const table = sheet.tables.add(`A1:O${lastRow}`, true);
  table.name = tableName;
  table.style = "TableStyleMedium2";
  table.showBandedRows = true;
  table.showFilterButton = true;
  sheet.showGridLines = false;

  const output = await artifactTool.SpreadsheetFile.exportXlsx(workbook);
  await output.save(outputPath);
}


const sourcePath = path.resolve(requireArgument(process.argv[2], "source workbook"));
const specificationPath = path.resolve(
  requireArgument(process.argv[3], "specification JSON"),
);
const artifactToolEntry = path.resolve(
  requireArgument(process.argv[4], "artifact-tool entry"),
);

const fs = await import("node:fs/promises");
const specification = JSON.parse(
  await fs.readFile(specificationPath, "utf8"),
);

for (const output of specification.outputs) {
  await buildWorkbook({
    sourcePath,
    outputPath: path.resolve(output.path),
    artifactToolEntry,
    columns: specification.columns,
    rows: output.rows,
    tableName: output.table_name,
  });
  console.log(`[built] ${output.path}`);
}
