import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression.GBTRegressor

object SparkML {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .master("local[1]")
      .appName("Titanic Survival Prediction")
      .getOrCreate()

    // Specify the path to your CSV file
    val trainFilePath = "train.csv"
    val testFilePath = "test.csv"

    // read csv file
    val train_df = spark.read
      .format("csv")
      .option("header", "true")
      .option("inferSchema", "true")
      .load(trainFilePath)

    val test_df = spark.read.format("csv").option("header","true").option("inferSchema","true").load(testFilePath)

    //EDA these datasets
    train_df.printSchema()
    test_df.printSchema()
    train_df.describe().show()
    test_df.describe().show()

    //Feature Engineering
    val train_features = train_df.select("Pclass","Sex","Age","SibSp","Parch","Fare","Survived")
    val test_features = test_df.select("Pclass", "Sex", "Age", "SibSp", "Parch", "Fare")

    //adding one column called FamilyNumber that is the sum of Sib and parent, in this case
    //then clean the sib and parch two columns
    val train_features_with_family = train_features.withColumn("FamilyNumber", train_features("SibSp") + train_features("Parch"))
    var train_final = train_features_with_family.drop("SibSp", "Parch")
    val test_features_with_family = test_features.withColumn("FamilyNumber", test_features("SibSp") + test_features("Parch"))
    var test_final = test_features_with_family.drop("SibSp", "Parch")

    train_final = train_final.withColumn("Sex", when(train_final("Sex") === "male", 0).otherwise(1))
    test_final = test_final.withColumn("Sex", when(test_final("Sex") === "male", 0).otherwise(1))
    val columnsToFill = Seq("Age", "Fare")
    val fillValues = Map(
      "Age" -> 0,
      "Fare" -> 0.0
    )
    train_final = train_final.na.fill(fillValues)
    test_final = test_final.na.fill(fillValues)
    train_final.printSchema()
    test_final.printSchema()

    val assembler = new VectorAssembler()
      .setInputCols(Array("Pclass", "Sex", "Age", "FamilyNumber", "Fare"))
      .setOutputCol("features")

    val df3 = assembler.transform(train_final)


    val gbt = new GBTRegressor()
      .setLabelCol("Survived")
      .setFeaturesCol("features")
      .setMaxIter(10)

    val model = gbt.fit(df3)
    val predictions = model.transform(df3)
    predictions.select("prediction", "Survived").show(5)
    val thresholdedPredictions = predictions.withColumn("Survived_Prediction", when(col("prediction") >= 0.5, 1).otherwise(0))
    thresholdedPredictions.select("Survived_Prediction","Survived").show(5)
    val correctPredictions = thresholdedPredictions.filter((col("Survived_Prediction") === 1 && col("Survived") === 1) || (col("Survived_Prediction") === 0 && col("Survived") === 0))
    val accuracy = correctPredictions.count().toDouble / thresholdedPredictions.count()
    println(s"Accuracy: $accuracy")


    val df4 = assembler.transform(test_final)
    val predictions1 = model.transform(df4)
    val thresholdedPredictions1 = predictions1.withColumn("Survived_Prediction", when(col("prediction") >= 0.5, 1).otherwise(0))
    thresholdedPredictions1.select("Survived_Prediction","Pclass", "Sex", "Age", "FamilyNumber", "Fare").show(20)


    spark.stop()
  }
}
