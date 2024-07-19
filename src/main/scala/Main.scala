package com.my.googleplay

import org.apache.spark.sql.{SparkSession, functions => F}
import org.apache.spark.sql.types._
import org.apache.spark.sql.expressions.Window

object Main {
  def main(args: Array[String]): Unit = {
    println("Starting the Spark application...")
    val spark = SparkSession.builder
      .appName("Google Play Store Apps")
      .master("local[*]")
      .getOrCreate()

    import spark.implicits._

    // Define schemas
    println("Schemas: 3")
    println("Schemas: 2")
    println("Schemas: 1")

    val appsSchema = StructType(Array(
      StructField("App", StringType, true),
      StructField("Category", StringType, true),
      StructField("Rating", DoubleType, true),
      StructField("Reviews", StringType, true),
      StructField("Size", StringType, true),
      StructField("Installs", StringType, true),
      StructField("Type", StringType, true),
      StructField("Price", StringType, true),
      StructField("Content Rating", StringType, true),
      StructField("Genres", StringType, true),
      StructField("Last Updated", StringType, true),
      StructField("Current Ver", StringType, true),
      StructField("Android Ver", StringType, true)
    ))


    val reviewsSchema = StructType(Array(
      StructField("App", StringType, true),
      StructField("Translated_Review", StringType, true),
      StructField("Sentiment", StringType, true),
      StructField("Sentiment_Polarity", DoubleType, true),
      StructField("Sentiment_Subjectivity", DoubleType, true)
    ))

    println("DONE.")

    // Load datasets
    println("Loading Datasets: 3")
    println("Loading Datasets: 2")
    println("Loading Datasets: 1")

    val appsDF = spark.read
      .option("header", "true")
      .schema(appsSchema)
      .csv("googleplaystore.csv")

    val reviewsDF = spark.read
      .option("header", "true")
      .schema(reviewsSchema)
      .csv("googleplaystore_user_reviews.csv")

    println("DONE.")

    // Part 1
    println("Starting Part 1: 3")
    println("Starting Part 1: 2")
    println("Starting Part 1: 1")


    val df1 = reviewsDF.groupBy("App")
      .agg(F.avg(F.coalesce($"Sentiment_Polarity", F.lit(0))).as("Average_Sentiment_Polarity"))

    println("DONE.")

    // Part 2
    println("Starting Part 2: 3")
    println("Starting Part 2: 2")
    println("Starting Part 2: 1")

    val bestAppsDF = appsDF.na.drop(Seq("Rating")) // Remove rows with NaN values in Rating
      .filter($"Rating" >= 4.0)
      .orderBy($"Rating".desc)
    bestAppsDF.write
      .mode("overwrite")  // Add this line to overwrite existing file
      .option("delimiter", "§")
      .csv("best_apps1")

    println("DONE.")

    // Part 3
    println("Starting Part 3: 3")
    println("Starting Part 3: 2")
    println("Starting Part 3: 1")


    val appsDFWithReviews = appsDF
      .withColumn("Reviews", F.when($"Reviews".isNull, 0).otherwise($"Reviews"))
      .withColumn("Price", F.regexp_replace($"Price", "\\$", "").cast(DoubleType) * 0.9)
      .withColumn("Size", F.regexp_replace($"Size", "M", "").cast(DoubleType))
      .withColumn("Size", F.when($"Size".isNull, 0).otherwise($"Size"))
      .withColumn("Last_Updated", F.to_date($"Last Updated", "MMMM d, yyyy"))

    val windowSpec = Window.partitionBy("App").orderBy($"Reviews".desc)

    val df3 = appsDFWithReviews.withColumn("RowNum", F.row_number().over(windowSpec))
      .groupBy("App")
      .agg(
        F.collect_set("Category").as("Categories"),
        F.first("Rating").as("Rating"),
        F.first("Reviews").as("Reviews"),
        F.first("Size").as("Size"),
        F.first("Installs").as("Installs"),
        F.first("Type").as("Type"),
        F.first("Price").cast(DoubleType).as("Price"),
        F.first("Content Rating").as("Content_Rating"),
        F.collect_set("Genres").as("Genres"),
        F.first("Last Updated").as("Last_Updated"),
        F.first("Current Ver").as("Current_Version"),
        F.first("Android Ver").as("Minimum_Android_Version")
      )
      .withColumn("Price", F.round($"Price" * 0.9, 2))
      .withColumn("Last_Updated", F.to_date($"Last_Updated", "MMMM d, yyyy"))
      .withColumn("Size", F.regexp_replace($"Size", "M", "").cast(DoubleType))
      .withColumn("Size", F.when($"Size".isNull, 0).otherwise($"Size"))
      .withColumn("Categories", F.concat_ws(",", $"Categories"))
      .withColumn("Genres", F.concat_ws(",", $"Genres"))
      .drop("RowNum")

    df3.write
      .mode("overwrite")
      .option("header", "true")
      .csv("df3_export")

    println("DONE.")

    // Part 4
    println("Starting Part 4: 3")
    println("Starting Part 4: 2")
    println("Starting Part 4: 1")

    val finalDF = df3.join(df1, Seq("App"), "left")
    finalDF.write
      .mode("overwrite")
      .option("compression", "gzip")
      .parquet("googleplaystore_cleaned")

    println("DONE.")

    // Part 5
    println("Starting Part 5: 3")
    println("Starting Part 5: 2")
    println("Starting Part 5: 1")

    val df4 = finalDF.select(F.col("Genres"), F.col("Rating"), F.col("Average_Sentiment_Polarity"))
      .withColumn("Genres", F.explode(F.split(F.col("Genres"), ","))) // Split Genres back to array for explosion
      .groupBy("Genres")
      .agg(
        F.count("*").as("Count"),
        F.avg("Rating").as("Average_Rating"),
        F.avg("Average_Sentiment_Polarity").as("Average_Sentiment_Polarity")
      )

    df4.write
      .mode("overwrite")
      .option("compression", "gzip")
      .parquet("googleplaystore_metrics")

    println("DONE.")


    println("Stopping the Spark application...")
    spark.stop()
  }
}

