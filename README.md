# Predictive Maintenance Business Dashboard

An interactive dashboard that helps a factory reduce unexpected machine failures by analysing machine usage, sensor readings, maintenance history and failure records.

**Author:** Lalithya J & Vindhya V
**Live app:** 

## Business problem
A factory wants to reduce unexpected machine failures. This project answers:
1. Which components and machine models fail most?
2. Do sensor readings show warning signs before a failure?
3. How does the maintenance schedule relate to failures?
4. Which machines are at highest risk right now?

## Data sources
- **Microsoft Azure Predictive Maintenance dataset** (public, simulated): 100 machines, one year (2015) of hourly telemetry (voltage, rotation, pressure, vibration), plus error logs, maintenance records, failure records and machine details.
  Kaggle: https://www.kaggle.com/arnabbiswas1/microsoft-azure-predictive-maintenance
  Original source: Azure AI Gallery, "Predictive Maintenance Modeling Guide". License: not specified on Kaggle ("Unknown"). The data originates from Microsoft's Azure AI Gallery "Predictive Maintenance Modeling Guide" and is used here for educational purposes only, with credit to the original authors.
- **AI-generated synthetic columns** (created in this project, see below).

## AI-generated data disclosure
The Azure dataset has no temperature or operating-hours columns, so both were generated in Python (`notebooks/01_exploration.ipynb`):
- **temperature:** baseline of 70, plus a term linked to rotation speed, random noise, and a rise of up to 8 degrees in the 72 hours before each failure.
- **operating_hours:** machine age x 8,760 + hours elapsed in the telemetry year.

Because temperature was designed to rise before failures, its importance in the model reflects the synthetic design, not a real-world discovery. All other sensor columns are original data.

## Method
1. **Clean:** loaded five tables, checked for missing values, duplicates and impossible values (none found).
2. **Engineer:** added temperature, operating hours and a label for "failure within the next 24 hours".
3. **Analyse:** failures by component, model and age; sensor behaviour before failures; maintenance cycle; errors vs failures.
4. **Model:** Random Forest trained on data before 1 Sept 2015 and tested on the rest, using 24-hour average readings, age and operating hours. Class weighting was used because failures are rare (about 2% of readings).
5. **Dashboard:** Streamlit app with a model filter and six tabs.

## Key findings
- **Components:** comp2 causes the most failures (about 259 of 761); comp3 the fewest (about 131).
- **Models:** model1 fails most often per machine (about 11.8 failures per machine), roughly twice as often as model4 (about 5.7).
- **Age:** machines aged 14 or older average about 8 to 10 failures, against about 4 to 6.5 for younger machines.
- **Warning signs:** in the 24 hours before a failure, rotation drops about 6% while vibration (+6%), pressure (+4.5%) and voltage (+3%) rise. No single sensor is a reliable alarm on its own.
- **Maintenance:** maintenance follows a fixed 15-day cycle. Failures occur a median of 67 days after the failing component was last replaced.
- **Errors:** machines with more logged errors fail more (correlation 0.54).
- **Model:** on unseen data it caught 5,376 of 5,407 failure hours (recall 0.99) with precision 0.59 (3,755 false alarms).
- **Highest risk:** machines 15, 64 and 95 (risk 0.71, 0.52, 0.44), all 14+ years old.

## Business recommendations
1. Inspect machines 15, 64 and 95 first, and machine 90 (young, but its risk is elevated).
2. Prioritise machines aged 14+ years for earlier or more frequent maintenance.
3. Monitor model1 and model2 machines more closely.
4. Stock spares and plan inspections for comp2, the most failure-prone component.
5. Replace fixed 15-day visits with risk-based scheduling, combining error counts and sensor trends.
6. Watch rotation, vibration, pressure and voltage together, since single sensors give weak warnings.

## Limitations
- Simulated data covering one year only.
- Temperature and operating hours are synthetic, which makes model results look better than real-world performance would.
- The model is a demonstration of the pipeline, not a production system.

## How to run
```bash
git clone [your repo link]
cd [repo name]
python -m venv venv
venv\Scripts\activate        # Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```
The cleaned data is in `data/processed/`. To rebuild it, download the five CSV files into `data/raw/` and run the notebooks in order (`01_exploration`, `02_analysis`, `03_model`).

## Project structure
```
app.py                 Streamlit dashboard
notebooks/             exploration, analysis and model notebooks
data/processed/        cleaned data used by the dashboard
requirements.txt       libraries
```

## Screenshots
![Failures](screenshots/failures.png)
![Warning signs](screenshots/warning-signs.png)
![Maintenance](screenshots/maintenance.png)
![Risk scores](screenshots/risk.png)