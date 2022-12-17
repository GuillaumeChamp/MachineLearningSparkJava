package org.example;

import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.clustering.KMeans;
import org.apache.spark.ml.regression.LinearRegression;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;

public class MLProcess {

    protected static void process(Dataset<Row> cleaned){
        //Split
        Dataset<Row>[] split = cleaned.randomSplit(new double[]{0.7,0.3});
        Dataset<Row> training = split[0];
        Dataset<Row> test = split[1];
        //create the stages
        KMeans kmeans = new KMeans()
                .setFeaturesCol("features")
                .setK(2);
        LinearRegression lr = new LinearRegression()
                .setFeaturesCol("features")
                .setLabelCol("y")
                .setMaxIter(10)
                .setElasticNetParam(0.8);
        //Create the pipeline
        Pipeline pipeline = new Pipeline()
                .setStages(new PipelineStage[] {lr, kmeans});
        //Use the pipeline
        PipelineModel model = pipeline.fit(training);
        model.transform(test).show();
    }
}
