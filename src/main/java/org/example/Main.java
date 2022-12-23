package org.example;

import org.apache.spark.SparkConf;
import org.apache.spark.SparkContext;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import java.io.File;
import java.io.IOException;
import java.util.Objects;

public class Main {
    protected static String trainingPath;
    protected static String testingPath;
    private final static String schema = "java -jar application.jar trainingPath testingPath OutputPath";
    private final static String supportedFormat = "Supported format are : .csv .json";
    private static String trainingExtension;
    private static String testingExtension;
    protected static String outPath = "./";
    private static boolean local=false;

    //To run add the following arguments Path\To\Documents\BD\1998.csv C:\Path\To\Documents\BD\1998.csv
    public static void main(String[] args) {
        if (!handleArguments(args)) return;
        SparkConf conf = new SparkConf().setAppName("FlightDelayLearning")
                .set("spark.executor.memory", "8g")
                .set("spark.storage.memoryFraction","1")
                .set("rdd.compression","true")
                .set("spark.driver.memory","8g");
        if (local) conf = conf.setMaster("local[2]");
        try {
            new SparkContext(conf);
        }catch (Exception e){
            conf = conf.setMaster("local[2]");
            new SparkContext(conf);
        }

        SparkSession spark = SparkSession
                .builder()
                .appName("FlightDelayLearningSQL")
                .getOrCreate();

        try {
            Dataset<Row> cleaned = DataProcessing.process(spark,trainingPath,trainingExtension);
            Dataset<Row> test;
            if (Objects.equals(trainingPath, testingPath)) test = cleaned;
            else test = DataProcessing.process(spark,testingPath,testingExtension);
            MLProcess.process(cleaned, test);
        } catch (Exception e) {
            e.printStackTrace();
            spark.stop();
        }

        spark.stop();
    }

    /**
     * Handle the argument passed to the application
     * @param args all the arguments
     * @return false if an error occurred
     */
    private static boolean handleArguments(String[] args) {
        if (args.length<2){
            System.out.println("wrong parameters");
            System.out.println(schema);
            return false;
        }
        if (!args[0].endsWith(".csv") && !args[0].endsWith(".json")){
            System.out.println("Training set in a unsupported format. " + supportedFormat);
            return false;
        }
        trainingPath = args[0];
        if (!args[1].endsWith(".csv")&& !args[1].endsWith(".json")){
            System.out.println("Testing set in a unsupported format. " + supportedFormat);
            return false;
        }
        //Extract extension
        String[] tamp = args[0].split("\\.");
        trainingExtension= tamp[tamp.length-1];
        tamp = args[1].split("\\.");
        testingExtension= tamp[tamp.length-1];
        if (args.length>=3)
            if (new File(args[2]).isDirectory()) outPath = args[2];
        if (args.length>=4) local=args[3].equals("local");
        try {
            createLog();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        testingPath = args[1];
        if (args[1].equals(args[0])){
            MyLog.log("You will use the same set for training and testing. This is not recommended, but it will work.");
        }
        return true;
    }

    private static void createLog() throws IOException {
        MyLog.init(outPath);
    }
}