package org.example;

import org.apache.spark.SparkConf;
import org.apache.spark.SparkContext;
import org.apache.spark.sql.SparkSession;

public class Main {
    protected static String trainingPath;
    protected static String testingPath;
    private final static String schema = "java -jar application.jar trainingPath testingPath OutputPath";
    private final static String supportedFormat = "Supported format are : .csv";
    private static String trainingExtension;

    //To run add the following arguments Path\To\Documents\BD\1998.csv C:\Path\To\Documents\BD\1998.csv
    public static void main(String[] args) {
        if (!handleArguments(args)) return;
        SparkConf conf = new SparkConf().setAppName("FlightDelayLearning").setMaster("local[2]").set("spark.executor.memory", "1g");
        new SparkContext(conf);
        SparkSession spark = SparkSession
                .builder()
                .appName("Java Spark SQL basic example")
                .config("spark.some.config.option", "some-value")
                .getOrCreate();

        try {
            DataProcessing.process(spark,trainingPath,trainingExtension);
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
        if (!args[0].endsWith(".csv")){
            System.out.println("training set in non supported format. " + supportedFormat);
            return false;
        }
        trainingPath = args[0];
        if (!args[1].endsWith(".csv")){
            System.out.println("testing set in non supported format. " + supportedFormat);
            //Extract extension
            String[] tamp = args[1].split("\\.");
            trainingExtension= tamp[tamp.length-1];
            return false;
        }
        testingPath = args[1];
        if (args[1].equals(args[0])){
            System.out.println("you will use the same set for training and testing this is not recommended but work fine");
        }
        return true;
    }
}