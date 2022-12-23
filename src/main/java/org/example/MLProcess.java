package org.example;

import org.apache.spark.ml.Model;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.evaluation.RegressionEvaluator;
import org.apache.spark.ml.feature.*;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.regression.LinearRegression;
import org.apache.spark.ml.regression.RandomForestRegressor;
import org.apache.spark.ml.tuning.CrossValidator;
import org.apache.spark.ml.tuning.CrossValidatorModel;
import org.apache.spark.ml.tuning.ParamGridBuilder;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;

public class MLProcess {
    private static final String[] imputedCols = new String[]{"Month","DayofMonth","DayOfWeek","CRSElapsedTime","DepDelay","TaxiOut","CRSDepTime_i","UniqueCarrier_i","FlightNum_i","TailNum_i","Origin_i","Dest_i"};
    private static final String[] column_cat = new String[]{"CRSDepTime_cat","UniqueCarrier","FlightNum","TailNum","Origin","Dest"};
    private static final String[] indexed = new String[]{"CRSDepTime_i","UniqueCarrier_i","FlightNum_i","TailNum_i","Origin_i","Dest_i"};
    private static final String[] encoded = new String[]{"CRSDepTime_e","UniqueCarrier_e","FlightNum_e","TailNum_e","Origin_e","Dest_e"};
    private static final String[] encoded_assembled = new String[]{"CRSDepTime_e","UniqueCarrier_e","FlightNum_e","TailNum_e","Origin_e","Dest_e","Month","DayofMonth","DayOfWeek","CRSElapsedTime","DepDelay","TaxiOut"};

    protected static void process(Dataset<Row> cleaned, Dataset<Row> test) {
        //create the stages of the pipeline
        Imputer imputer = new Imputer()
                .setStrategy("mode")
                .setInputCols(imputedCols)
                .setOutputCols(imputedCols);
        StringIndexer indexer = new StringIndexer()
                .setInputCols(column_cat)
                .setOutputCols(indexed)
                .setHandleInvalid("keep");
        OneHotEncoder encoder = new OneHotEncoder()
                .setInputCols(indexed)
                .setOutputCols(encoded)
                .setHandleInvalid("keep");
        VectorAssembler assembler = new VectorAssembler() //Numerical
                .setInputCols(encoded_assembled)
                .setOutputCol("features");
        Normalizer normalizer = new Normalizer()
                .setInputCol("features")
                .setOutputCol("NormalizedFeatures"); //p=2
        LinearRegression lr = new LinearRegression()
                .setMaxIter(10)
                .setRegParam(0.3)
                .setElasticNetParam(0.8)
                .setFeaturesCol("NormalizedFeatures")
                .setLabelCol("ArrDelay")
                .setPredictionCol("prediction_lr");
        RandomForestRegressor rm = new RandomForestRegressor()
                .setFeaturesCol("NormalizedFeatures")
                .setLabelCol("ArrDelay")
                .setPredictionCol("prediction_rm")
                .setMaxDepth(3)
                .setNumTrees(20);
        Pipeline pipeline = new Pipeline().setStages(new PipelineStage[]{
                indexer
                , imputer
                , encoder
                , assembler
                , normalizer
                , lr
        });
        Pipeline pipeline1 = new Pipeline().setStages(new PipelineStage[]{
                indexer
                , imputer
                , encoder
                , assembler
                , normalizer
                , rm
        });

        //Cross validation
        ParamMap[] paramGrid = new ParamGridBuilder()
                .addGrid(lr.regParam(), new double[]{0, 1, 0.3, 0.03})
                .build();

        ParamMap[] paramGrid_rm = new ParamGridBuilder()
                .addGrid(rm.maxDepth(), new int[]{2, 3, 4})
                .addGrid(rm.numTrees(), new int[]{20, 25})
                .build();

        CrossValidator cv_lr = new CrossValidator()
                .setEstimator(pipeline)
                .setEvaluator(new RegressionEvaluator()
                        .setLabelCol("ArrDelay")
                        .setPredictionCol("prediction_lr")
                        .setMetricName("rmse"))
                .setEstimatorParamMaps(paramGrid)
                .setNumFolds(5)  // Use 3+ in practice
                .setParallelism(2);  // Evaluate up to 2 parameter settings in parallel

        CrossValidator cv_rm = new CrossValidator()
                .setEstimator(pipeline1)
                .setEvaluator(new RegressionEvaluator()
                        .setLabelCol("ArrDelay")
                        .setPredictionCol("prediction_rm")
                        .setMetricName("rmse"))
                .setEstimatorParamMaps(paramGrid_rm)
                .setNumFolds(5)  // Use 3+ in practice
                .setParallelism(2);  // Evaluate up to 2 parameter settings in parallel


        // Run cross-validation, and choose the best set of parameters.
        CrossValidatorModel cvModel_lr = cv_lr.fit(cleaned);

        Dataset<Row> predictions_lr = cvModel_lr.transform(test);

        RegressionEvaluator evaluatorRMSE_lr = new RegressionEvaluator()
                .setLabelCol("ArrDelay")
                .setPredictionCol("prediction_lr")
                .setMetricName("rmse");
        double RMSE_lr = evaluatorRMSE_lr.evaluate(predictions_lr);


        Model<?> model_lr = cvModel_lr.bestModel();

        try {
            MyLog.log("Linear regression best RMSE:");
            MyLog.log(Double.toString(RMSE_lr));
            MyLog.log("Linear regression chosen parameters:");
            MyLog.log("regParam:");
            MyLog.log(String.valueOf(model_lr.getParam("regParam")));
        }catch (Exception ignored){}


        predictions_lr.select("ArrDelay", "prediction_lr").show(10);
        if (Main.randomTree) {
            CrossValidatorModel cvModel_rm = cv_rm.fit(cleaned);
            Dataset<Row> predictions_rm = cvModel_rm.transform(test);
            RegressionEvaluator evaluatorRMSE_rm = new RegressionEvaluator()
                    .setLabelCol("ArrDelay")
                    .setPredictionCol("prediction_rm")
                    .setMetricName("rmse");
            double RMSE_rf = evaluatorRMSE_rm.evaluate(predictions_rm);
            Model<?> model_rm = cvModel_rm.bestModel();
            MyLog.log("Random Forest best RMSE:");
            MyLog.log(String.valueOf(RMSE_rf));
            try {
                MyLog.log("Random Forest chosen parameters:");
                MyLog.log("numTrees:");
                MyLog.log(String.valueOf(model_rm.getParam("numTrees")));
                MyLog.log("maxDepth:");
                MyLog.log(String.valueOf(model_rm.getParam("maxDepth")));
            }catch (Exception ignored){}
            if (RMSE_lr <= RMSE_rf) {
                predictions_lr.write().format("csv").save(Main.outPath + "predict.csv");
            } else {
                predictions_rm.write().format("csv").save(Main.outPath + "predict.csv");
            }
        } else {

            predictions_lr.write().format("csv").save(Main.outPath + "predict.csv");
        }
    }

}
