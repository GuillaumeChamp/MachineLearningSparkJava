package org.example;

import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.evaluation.RegressionEvaluator;
import org.apache.spark.ml.feature.Imputer;
import org.apache.spark.ml.feature.Normalizer;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.regression.LinearRegression;
import org.apache.spark.ml.regression.RandomForestRegressor;
import org.apache.spark.ml.tuning.CrossValidator;
import org.apache.spark.ml.tuning.CrossValidatorModel;
import org.apache.spark.ml.tuning.ParamGridBuilder;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;

public class MLProcess {
    private static final String[] imputedCols = new String[]{"ArrDelay","DepDelay","TaxiOut"};
    private static final String[] imputedColsO = new String[]{"ArrDelayImputed","DepDelayImputed","TaxiOutImputed"};
    private static final String[] assembled = new String[]{"Month","DayofMonth","DayOfWeek"};
    protected static CrossValidatorModel process(Dataset<Row> cleaned){
        //create the stages
        Imputer imputer = new Imputer()
                .setStrategy("mean")
                .setInputCols(imputedCols)
                .setOutputCols(imputedColsO);
        Normalizer normalizer = new Normalizer()
                .setInputCol("features")
                .setOutputCol("NormalizedFeatures"); //p=2
        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(assembled)
                .setOutputCol("features");
        LinearRegression lr = new LinearRegression()
                .setMaxIter(10)
                .setRegParam(0.3)
                .setElasticNetParam(0.8)
                .setLabelCol("ArrDelay");
        RandomForestRegressor rm = new RandomForestRegressor()
                .setLabelCol("prediction");
        //Create the pipeline
        Pipeline pipeline = new Pipeline()
                .setStages(new PipelineStage[] {imputer
                        ,assembler
                        ,normalizer
                        ,lr
                        //,rm
                });
        
        //Cross validation
        ParamMap[] paramGrid = new ParamGridBuilder()
                .addGrid(lr.regParam(), new double[] {0.1, 0.01})
                .build();

        CrossValidator cv = new CrossValidator()
                .setEstimator(pipeline)
                .setEvaluator(new RegressionEvaluator().setLabelCol("FlightNum"))
                .setEstimatorParamMaps(paramGrid)
                .setNumFolds(10)  // Use 3+ in practice
                .setParallelism(2);  // Evaluate up to 2 parameter settings in parallel

        // Run cross-validation, and choose the best set of parameters.
        CrossValidatorModel cvModel = cv.fit(cleaned);
        //Use the pipeline
        cvModel.transform(cleaned).show();
        return cvModel;
    }
}
