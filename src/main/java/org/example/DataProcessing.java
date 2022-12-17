package org.example;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import java.util.Arrays;
import java.util.List;

public class DataProcessing {
    private static final List<String> dropped = Arrays.asList("DepTime","Year","ArrTime","CRSArrTime","ActualElapsedTime","AirTime","Distance","TaxiIn","Cancelled","CancellationCode","Diverted","CarrierDelay","WeatherDelay","NASDelay","SecurityDelay","LateAircraftDelay");
    private static final List<String> toFilter = Arrays.asList("DepDelay","ArrDelay","CRSElapsedTime");

    static void process(SparkSession spark, String trainingPath, String trainingExtension) throws Exception {
        Dataset<Row> df = spark.read().option("inferSchema", "true").option("header", "true").csv(trainingPath);
        //spark.read().format(trainingExtension).option("inferSchema",true).option("header", "true").load(trainingPath);

        //counting to remove
        long numberOfRows= df.count();
        long nullValues;
        /*
        //TODO : Too heavy to optimise
        for (String column : df.columns()){
            nullValues = df.where(df.col(column).equalTo("NA")).count();
            if (nullValues>0.4*numberOfRows && !dropped.contains(column)) dropped.add(column);
            System.out.println("with your dataset "+ column +" is also dropped because to many missing values");
        }

         */
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
        filtered = filtered.withColumn("CRSDepTime",filtered.col("CRSDepTime").divide(20).cast("int").multiply(20).cast("int"));

        filtered.printSchema();
        df.show(10);
        filtered.show(10);

        if (filtered.count()<1300) {
            System.out.println("After cleaning less than 1300 data remains");
            throw new Exception("Not Enough Data Remaining");
        }

    }
}
