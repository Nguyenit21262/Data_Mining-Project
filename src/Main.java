import Pre_processing.DataPreprocessing;
import model_evaluation.ModelEvaluator;
import model_training.ModelTrainer;
import weka.core.Instances;


public class Main {
    public static void main(String[] args) throws Exception {

        DataPreprocessing.preprocess();
       
        // Train and Evaluate J48
        Instances trainDataJ48 = ModelTrainer.trainJ48("C:\\Users\\admin\\eclipse-workspace\\DataMining-FinalProject\\src\\Data\\cleaned_data.arff");
        ModelEvaluator.evaluateJ48(trainDataJ48);
        
        Instances trainDataImproved = ModelTrainer.trainJ48WithNaiveBayesImprovement("C:\\Users\\admin\\eclipse-workspace\\DataMining-FinalProject\\src\\Data\\cleaned_data.arff");
        ModelEvaluator.evaluateJ48(trainDataImproved); 
       
    }
}
