package org.example;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import java.util.Arrays;
import java.util.List;

public class DataProcessing {
    private static final List<String> dropped = Arrays.asList("ArrTime", "ActualElapsedTime", "AirTime", "TaxiIn", "TaxiOut", "Cancelled", "CancellationCode", "Diverted", "CarrierDelay", "WeatherDelay", "NASDelay", "SecurityDelay", "LateAircraftDelay");
    private static final List<String> toFilter = Arrays.asList("DepTime","DepDelay","ArrDelay","DepDelay","CRSElapsedTime");

    static void process(SparkSession spark, String trainingPath, String trainingExtension){
        Dataset<Row> df = spark.read().option("inferSchema", "true").option("header", "true").csv(trainingPath);
        //spark.read().format(trainingExtension).option("inferSchema",true).option("header", "true").load(trainingPath);
        df.show(2);
        df.printSchema();
        //dropping column
        Dataset<Row> filtered = df;
        for(String s : dropped){
            filtered = filtered.drop(s);
        }
        //filtering + casting
        for(String s : toFilter){
            filtered=filtered.where(s+" > -1");
            filtered = filtered.withColumn(s,filtered.col(s).cast("int"));
        }
        filtered.printSchema();
        filtered.show(2);

    }
}
