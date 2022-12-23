package org.example;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import java.util.Arrays;
import java.util.List;

public class DataProcessing {
    private static final List<String> dropped = Arrays.asList("DepTime","Year","ArrTime","CRSArrTime","ActualElapsedTime","AirTime","Distance","TaxiIn","Cancelled","CancellationCode","Diverted","CarrierDelay","WeatherDelay","NASDelay","SecurityDelay","LateAircraftDelay");
    private static final List<String> toFilter = Arrays.asList("DepDelay","ArrDelay","CRSElapsedTime");
    private static final List<String> toCheck = Arrays.asList("TaxiOut", "TailNum");

    protected static Dataset<Row> process(SparkSession spark, String trainingPath, String trainingExtension) throws Exception {
        Dataset<Row> df = null;
        if (trainingExtension.equals("csv")) df = spark.read().option("inferSchema", "true").option("header", "true").csv(trainingPath);
        //spark.read().format(trainingExtension).option("inferSchema",true).option("header", "true").load(trainingPath);

        //counting to remove
        long numberOfRows= df.count();
        long nullValues;
        for (String column : toCheck){
            nullValues = df.where(df.col(column).equalTo("NA")).count();
            if (nullValues>0.4*numberOfRows && !dropped.contains(column)) dropped.add(column);
            Logger.log("with your dataset "+ column +" is also dropped because to many missing values");
        }

        //Before dropping Cancelled, remove rows with flights that didn't arrive
        df = df.where("Cancelled == '0'");

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
        //Categorization of CRSDepTime
        filtered = filtered.withColumn("CRSDepTime_cat",filtered.col("CRSDepTime").divide(20).cast("int").multiply(20).cast("string"));

        filtered.printSchema();
        filtered.summary();

        if (filtered.count()<1300) {
            Logger.log("After cleaning less than 1300 data remains");
            throw new Exception("Not Enough Data Remaining");
        }
        return filtered;
    }
}
