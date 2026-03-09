# San Diego University Submarkets Housing Analysis

This project evaluates housing investment opportunities in residential submarkets near three San Diego universities:

- **SDSU** (College Area, ZIP 92115)
- **UCSD** (University City / UTC, ZIP 92122)
- **USD** (Linda Vista / Morena, ZIP 92110)

The analysis was built as an **MGT 188 final project** and frames the question from a **real estate / REIT investment** perspective.

## Project question

**Which university-adjacent housing submarket in San Diego looks most attractive for a housing-focused real estate investment strategy?**

The project compares each university submarket using:

- total enrollment as a demand proxy
- planned student housing expansion
- average home values
- rent-to-value ratio
- transit accessibility
- FHFA house price index history for a real-data extension

## Main result

The screening model ranks the three submarkets as follows:

1. **SDSU** - strongest overall current investment balance
2. **UCSD** - strongest long-term growth and expansion story
3. **USD** - useful benchmark, but weaker direct investment signals

### Key takeaways

- **SDSU ranked first** with an investment score of **0.932**.
- **UCSD ranked second** with an investment score of **0.600**.
- **USD ranked third** with an investment score of **0.168**.
- SDSU had the **lowest average home value** among the three markets at **$835,518** and the **highest rent-to-value ratio** at **6.18%**.
- UCSD showed the **largest housing expansion pipeline**, with roughly **18,910 net new beds** in the project dataset.
- In the FHFA extension, the latest annual change in the prepared file shows **UCSD at 37.35%**, **SDSU at 5.88%**, and **USD at 2.86%**.

## Repository structure

```text
.
├── analysis.py
├── README.md
├── TEAM_ROLES.md
├── requirements.txt
├── data/
│   ├── fhfa_2022-2024.csv
│   ├── fhfa_full_history.csv
│   ├── fhfa_historical_data.xlsx
│   ├── resource_links.csv
│   ├── starter_data.csv
│   └── university_submarkets.xlsx
├── outputs/
│   ├── figures/
│   └── tables/
└── presentations/
    ├── final-project-rough.pptx
    └── final-presentation-group-7.pdf
