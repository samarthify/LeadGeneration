# CSV Formats

## Input CSV Format (For Uploads)

Users should upload a CSV with exactly **2 columns**:

```csv
Company Name,Website
Sony,https://www.sony.com
Toyota,https://www.toyota.com
Microsoft,https://www.microsoft.com
```

### Input Format Rules:
- **Column 1**: Company Name (required)
- **Column 2**: Website (optional, can be empty)
- **Header row**: Include the header "Company Name,Website"
- **Maximum**: 100 companies per upload

### Example Input CSV:
```csv
Company Name,Website
Sony,https://www.sony.com
Toyota,https://www.toyota.com
Microsoft,https://www.microsoft.com
Apple,https://www.apple.com
Google,https://www.google.com
```

---

## Output CSV Format (Generated Results)

The app generates a CSV with exactly **8 columns** in Japanese:

```csv
会社名, 部署, 役職, 姓, 名, 姓（小文字ローマ字）, 名（小文字ローマ字）, メールアドレスに使用される可能性が高いドメイン
```

### Column Descriptions:

| Column | Japanese | English | Description |
|--------|----------|---------|-------------|
| 1 | 会社名 | Company name | The company name |
| 2 | 部署 | Department | Employee's department/division |
| 3 | 役職 | Job title | Employee's job title/position |
| 4 | 姓 | Last name | Employee's last name (native script) |
| 5 | 名 | First name | Employee's first name (native script) |
| 6 | 姓（小文字ローマ字） | Last name (lowercase Roman letters) | Romanized last name in lowercase |
| 7 | 名（小文字ローマ字） | First name (lowercase Roman letters) | Romanized first name in lowercase |
| 8 | メールアドレスに使用される可能性が高いドメイン | Domain likely to be used in email addresses | Company's email domain |

### Example Output CSV:
```csv
会社名, 部署, 役職, 姓, 名, 姓（小文字ローマ字）, 名（小文字ローマ字）, メールアドレスに使用される可能性が高いドメイン
Sony, Corporate Executive, President and CEO, 吉田, 憲一郎, yoshida, kenichiro, sony.com
Sony, Music Publishing, Chairman and CEO, プラット, ジョン, platt, jon, sony.com
Sony, Financial Group, President and CEO, 岡, 正志, oka, masashi, sony.com
Sony, Corporate Communications, Senior Vice President, ローソン, ロバート, lawson, robert, sony.com
Sony, Interactive Entertainment, President and CEO, 西野, 英明, nishino, hideaki, sony.com
```

### Output Format Rules:

#### ✅ **Correct Format**
- **Commas separate fields** - Use commas as delimiters
- **No commas within fields** - Replace commas in job titles with semicolons or spaces
- **Lowercase romanized names** - All romanized names should be lowercase
- **Complete names** - Both first and last names should be provided
- **Consistent domain** - Same email domain for all employees

#### ❌ **Common Issues to Avoid**
- **Missing names** - Don't include entries without complete names
- **Duplicate entries** - Each person should appear only once
- **Generic titles** - Avoid "Administrative Assistant" without specific names
- **Commas in fields** - Don't use commas within job titles or department names
- **Mixed case** - Romanized names should be lowercase

### Sample Valid Entry:
```csv
Sony, Technology Strategy, Senior Vice President, 松本, 義則, matsumoto, yoshinori, sony.com
```

### Sample Invalid Entry:
```csv
Sony, Corporate Communications, Administrative Assistant, , , , , sony.com
```
*This is invalid because it has missing names.*

### Quality Guidelines:
1. **Complete Information** - Every entry should have both first and last names
2. **Accurate Titles** - Job titles should be specific and accurate
3. **Proper Romanization** - Names should be correctly romanized
4. **Consistent Format** - All entries should follow the same format
5. **No Duplicates** - Each person should appear only once
6. **Reasonable Quantity** - Aim for 20-40 quality entries, not hundreds of duplicates 