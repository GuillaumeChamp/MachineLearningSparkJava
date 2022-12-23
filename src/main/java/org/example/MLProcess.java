package org.example;

import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.evaluation.RegressionEvaluator;
import org.apache.spark.ml.feature.Imputer;
import org.apache.spark.ml.feature.Normalizer;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.OneHotEncoder;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.regression.LinearRegression;
import org.apache.spark.ml.regression.LinearRegressionModel;
import org.apache.spark.ml.regression.RandomForestRegressor;
import org.apache.spark.ml.regression.RandomForestRegressionModel;
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

    protected static CrossValidatorModel process(Dataset<Row> cleaned, Dataset<Row> test){

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
                        //, rm
                });


        //Cross validation
        ParamMap[] paramGrid = new ParamGridBuilder()
                .addGrid(lr.regParam(), new double[] {0,1, 0.3, 0.03})
                .build();

        ParamMap[] paramGrid_rm = new ParamGridBuilder()
                .addGrid(rm.maxDepth(), new int[] {2, 3, 4})
                .addGrid(rm.numTrees(), new int[] {20, 25})
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
                .setEstimator(pipeline)
                .setEvaluator(new RegressionEvaluator()
                        .setLabelCol("ArrDelay")
                        .setPredictionCol("prediction_rm")
                        .setMetricName("rmse"))
                .setEstimatorParamMaps(paramGrid_rm)
                .setNumFolds(5)  // Use 3+ in practice
                .setParallelism(2);  // Evaluate up to 2 parameter settings in parallel

        // Run cross-validation, and choose the best set of parameters.
        CrossValidatorModel cvModel_lr = cv_lr.fit(cleaned);
        CrossValidatorModel cvModel_rm = cv_rm.fit(cleaned);

        Dataset<Row> predictions_lr = cvModel_lr.transform(test);
        Dataset<Row> predictions_rm = cvModel_rm.transform(test);
        RegressionEvaluator evaluatorRMSE_lr = new RegressionEvaluator()
                .setLabelCol("ArrDelay")
                .setPredictionCol("prediction_lr")
                .setMetricName("rmse");
        Double RMSE_lr = evaluatorRMSE_lr.evaluate(predictions_lr);

        RegressionEvaluator evaluatorRMSE_rm = new RegressionEvaluator()
                .setLabelCol("ArrDelay")
                .setPredictionCol("prediction_rm")
                .setMetricName("rmse");
        Double RMSE_rf = evaluatorRMSE_rm.evaluate(predictions_rm);

        LinearRegressionModel model_lr = (LinearRegressionModel)cvModel_lr.bestModel();
        RandomForestRegressionModel model_rm = (RandomForestRegressionModel)cvModel_lr.bestModel();

        System.out.println("Linear regression best RMSE:");
        System.out.println(RMSE_lr);
        System.out.println("Linear regression coefficients:");
        System.out.println(model_lr.coefficients());
        System.out.println("Linear regression chosen parameters:");
        System.out.println("regParam:");
        System.out.println(model_lr.getRegParam());
        System.out.println();

        System.out.println("Random Forest best RMSE:");
        System.out.println(RMSE_rf);
        System.out.println("Random Forest chosen parameters:");
        System.out.println("numTrees:");
        System.out.println(model_rm.getNumTrees());
        System.out.println("maxDepth:");
        System.out.println(model_rm.getMaxDepth());
        System.out.println();

        predictions_lr.select("ArrDelay", "prediction_lr").show(10);
        predictions_rm.select("ArrDelay", "prediction_rm").show(10);

        if (RMSE_lr <= RMSE_rf) {
            return cvModel_lr;
        }
        else {
            return cvModel_rm;
        }
    }


}
