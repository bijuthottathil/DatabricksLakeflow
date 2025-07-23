import dlt
from pyspark.sql.functions import col, row_number
from pyspark.sql.window import Window


@dlt.table(
    name="bijucatalog.bijugoldschema.high_risk_patients",
    comment="Latest vitals for patients exceeding critical thresholds"
)
def high_risk_patients():
    # Read cleansed device data and patient demographics
    dev = dlt.read("bijucatalog.bijusilverschema.silver_medical_devices")
    pat = dlt.read("bijucatalog.bijusilverschema.silver_patients")
    
    # Define emergency thresholds
    HR_THRESHOLD   = 120
    SYS_THRESHOLD  = 140
    DIA_THRESHOLD  =  90
    
    # Window spec to get the latest record per patient
    w = Window.partitionBy("patient_id").orderBy(col("timestamp").desc())
    
    # Pick the latest reading for each patient
    latest = (
        dev
        .withColumn("rn", row_number().over(w))
        .filter(col("rn") == 1)
        .drop("rn")
    )
    
    # Filter to only critical cases
    critical = latest.filter(
        (col("heart_rate") > HR_THRESHOLD) |
        (col("systolic")   > SYS_THRESHOLD) |
        (col("diastolic")  > DIA_THRESHOLD)
    )
    
    # Join with demographics and select summary fields
    return (
        critical
        .join(pat, on="patient_id", how="left")
        .select(
            col("patient_id"),
            col("name"),
            col("gender"),
            col("timestamp").alias("latest_timestamp"),
            col("heart_rate"),
            col("systolic"),
            col("diastolic")
        )
    )