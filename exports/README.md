# Exports Directory

This directory is used to store exported search results from the GitHub Stars Search tool.

## Contents

When you run the export functionality, files will be saved here with timestamps in their filenames. For example:

- `search_results_20250520_123456.json` - JSON format
- `search_results_20250520_123456.csv` - CSV format
- `search_results_20250520_123456.md` - Markdown format
- `search_results_20250520_123456.html` - HTML format
- `search_results_20250520_123456.xml` - XML format

## Usage

You can generate exports by running the export example:

```bash
python examples/export_results.py
```

Or by specifying a specific format:

```bash
python examples/export_results.py --format json
```

See the [examples README](../examples/README.md) for more information on the export functionality.

## Viewing Exports

- JSON files can be viewed in any text editor or JSON viewer
- CSV files can be opened in spreadsheet applications like Excel or Google Sheets
- Markdown files can be viewed in any Markdown viewer or text editor
- HTML files can be opened in any web browser
- XML files can be viewed in any text editor or XML viewer

## Note

This directory is excluded from version control in the `.gitignore` file to avoid committing large data files to the repository.
