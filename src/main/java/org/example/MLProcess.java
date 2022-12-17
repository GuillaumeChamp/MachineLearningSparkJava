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
import org.apache.spark.ml.regression.RandomForestRegressor;
import org.apache.spark.ml.tuning.CrossValidator;
import org.apache.spark.ml.tuning.CrossValidatorModel;
import org.apache.spark.ml.tuning.ParamGridBuilder;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;

public class MLProcess {
    private static final String[] imputedCols = new String[]{"Month","DayofMonth","DayOfWeek","CRSElapsedTime","DepDelay","TaxiOut","CRSDepTime_i","UniqueCarrier_i","FlightNum_i","TailNum_i","Origin_i","Dest_i"};
    private static final String[] assembled = new String[]{"Month","DayofMonth","DayOfWeek","CRSElapsedTime","DepDelay","TaxiOut"};
    private static final String[] column_cat = new String[]{"CRSDepTime","UniqueCarrier","FlightNum","TailNum","Origin","Dest"};
    private static final String[] indexed = new String[]{"CRSDepTime_i","UniqueCarrier_i","FlightNum_i","TailNum_i","Origin_i","Dest_i"};
    private static final String[] encoded = new String[]{"CRSDepTime_e","UniqueCarrier_e","FlightNum_e","TailNum_e","Origin_e","Dest_e"};
    protected static CrossValidatorModel process(Dataset<Row> cleaned){
        //create the stages
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
                .setInputCols(assembled)
                .setOutputCol("features");
        VectorAssembler assembler2 = new VectorAssembler() // Categorical (One-Hot Encoded)
                .setInputCols(encoded)
                .setOutputCol("features2");
        VectorAssembler assembler3 = new VectorAssembler() // All
                .setInputCols(new String[]{"features", "features2"})
                .setOutputCol("features3");
        Normalizer normalizer = new Normalizer()
                .setInputCol("features3")
                .setOutputCol("NormalizedFeatures"); //p=2
        LinearRegression lr = new LinearRegression()
                .setMaxIter(10)
                .setRegParam(0.3)
                .setElasticNetParam(0.8)
                .setFeaturesCol("NormalizedFeatures")
                .setLabelCol("ArrDelay");
        RandomForestRegressor rm = new RandomForestRegressor()
                .setLabelCol("prediction");
        //Create the pipeline
        Pipeline pipeline = new Pipeline()
                .setStages(new PipelineStage[] {
                        indexer
                        ,imputer
                        ,encoder
                        ,assembler
                        ,assembler2
                        ,assembler3
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
                .setEvaluator(new RegressionEvaluator().setLabelCol("ArrDelay"))
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
