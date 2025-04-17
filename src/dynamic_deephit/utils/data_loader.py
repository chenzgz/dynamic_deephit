import pandas as pd
import pkg_resources

def load_test_data():
    """
    Load the training dataset.
    Description:
    The data comes from a kidney transplant failure study involving 407 patients who underwent kidney transplant surgeries at the Catholic University of Leuven in Belgium between January 1983 and August 2000.
    ID: Patients identifier; in total there are 407 patients.;
    time: Time-to-event data for renal graft failure or maximum follow up time, with transplant date as the time origin (years).
    status: Censoring indicator (1 = graft failure and 0 = censored).
    age: Patient's age (in years/10).
    weight: Patient's weight (in kg/10).
    sex: Gender of patient(1 = male and 0 = femal).
    hema: Hematocrit, recorded as percentage (0.1\%) of the ratio of the volume of red blood cells to the total volume of blood.
    GFR: Measured as 10 ml/min/1.73 m².
    Proteinuria: Measured as percentage 1 g/24 hours.
    yearse: Observation time.
    Source:
        - references: Rizopoulos D, Ghosh P. A Bayesian semiparametric multivariate joint model for multiple longitudinal outcomes and a time‐to‐event. Statistics in Medicine 2011;30:1366–80. https://doi.org/10.1002/sim.4205.
    """
    csv_path = pkg_resources.resource_filename('dynamic_deephit', 'data/renal.csv')
    return pd.read_csv(csv_path)

if __name__ == "__main__":
    df = load_test_data()
    print(df.head())