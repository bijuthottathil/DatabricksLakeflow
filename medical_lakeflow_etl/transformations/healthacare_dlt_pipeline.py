import dlt
from pyspark.sql.functions import col, window, avg,datediff,current_date,round
from pyspark.sql.types import  StructType, StructField, StringType, TimestampType, IntegerType


# ----------------------------
# 1. Bronze Layer: Ingest raw CSV data
# ----------------------------

@dlt.table(
    name="bijucatalog.bijubronzeschema.bronze_medical_devices",
    comment="Raw medical device data"
)
def bronze_medical_devices():
    return (
        spark.read.option("header", True)
        .schema(
            StructType([
                StructField("device_id", StringType()),
                StructField("patient_id", StringType()),
                StructField("timestamp", TimestampType()),
                StructField("heart_rate", IntegerType()),
                StructField("blood_pressure", StringType()),
            ])
        )
        .csv("s3://databricksbijubucketnew/raw/medica_devices/")
    )
    
@dlt.table(
    name="bijucatalog.bijubronzeschema.bronze_patients",
    comment="Raw patient data"
)
def bronze_patients():
    return (
        spark.read.option("header", True)
        .schema(
            StructType([
                StructField("patient_id", StringType()),
                StructField("name", StringType()),
                StructField("dob", StringType()),
                StructField("gender", StringType()),                 
               ])
        )
        .csv("s3://databricksbijubucketnew/raw/patient_records/")
    )


# ----------------------------
# 2. Silver Layer: Cleanse data
# ------------------------------
@dlt.table(
    name="bijucatalog.bijusilverschema.silver_medical_devices",
    comment="Cleanse medical device data"
)
@dlt.expect("valid_heart_rate", "heart_rate > 0 and heart_rate < 200 and blood_pressure != 'null'")
def silver_medical_devices():
    return (
        dlt.read("bijucatalog.bijubronzeschema.bronze_medical_devices")
        .withColumn("systolic", col("blood_pressure").substr(1,3).cast(IntegerType()))
        .withColumn("diastolic", col("blood_pressure").substr(5,3).cast(IntegerType()))

    )

@dlt.table(
    name="bijucatalog.bijusilverschema.silver_patients",
    comment="Cleanse patient data"
)
def silver_patients():
    return dlt.read("bijucatalog.bijubronzeschema.bronze_patients")

# ---------------------------------
# Gold Layer: Insights using sliding window
# ---------------------------------
@dlt.table(
    name="bijucatalog.bijugoldschema.gold_clinical_insights",
    comment="Insights using sliding window"
)
def gold_clinical_insights():
    devices=dlt.read("bijucatalog.bijusilverschema.silver_medical_devices")
    return (
        devices
        .groupBy(
            window(col("timestamp"), "30 minutes", "5 minutes"),
            col("patient_id")
        )
        .agg(avg(col("systolic")).alias("avg_systolic"),
            avg(col("heart_rate")).alias("avg_heart_rate"),
            avg(col("systolic")).alias("avg_diastolic"))
    )

# -----------------------------------------
# Clinical Alerts: avg_heart_rate > 100
# ------------------------------------------
@dlt.table(
    name="bijucatalog.bijugoldschema.gold_clinical_alerts",
    comment="Patients with avg heart rate > 100 over 30-min sliding windows"
)
def gold_clinical_alerts():
    devices = dlt.read("bijucatalog.bijusilverschema.silver_medical_devices")
    return (
        devices.groupBy(
            window(col("timestamp"), "30 minutes", "5 minutes"),
            col("heart_rate"),
            col("patient_id")
        )
        .agg(avg(col("heart_rate")).alias("avg_heart_rate"))
        .filter(col("avg_heart_rate") > 100)
    )

@dlt.table(
    name="bijucatalog.bijugoldschema.gold_compliance_report",
    comment="Compliance report on vitals grouped by gender and age group"   
)
def gold_compliance_report():
    device_df = dlt.read("bijucatalog.bijusilverschema.silver_medical_devices")
    patient_df = dlt.read("bijucatalog.bijusilverschema.silver_patients")
    joined_df = (
        device_df.join(patient_df, "patient_id", "inner")
        .withColumn("age", datediff(current_date(), col("dob")) / 365)
        .withColumn("age_group", (col("age").cast("int") / 10).cast("int") * 10)
    )
    return (
        joined_df.groupBy("gender", "age_group")
        .agg(
            round(avg("heart_rate"), 2).alias("avg_heart_rate"),
            round(avg("systolic"), 2).alias("avg_systolic"),
            round(avg("diastolic"), 2).alias("avg_diastolic")
        )
    )