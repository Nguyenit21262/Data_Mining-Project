package model_evaluation;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import java.util.Random;

public class ModelEvaluator {


    public static void evaluateJ48(Instances train_data) throws Exception {
    	
    	J48 tree = new J48();
        tree.buildClassifier(train_data);
        
        System.out.println("\n***** J48 Decision Tree ***");
        System.out.println(tree.toString());
        
        Evaluation evalJ48 = new Evaluation(train_data);
        evalJ48.crossValidateModel(tree, train_data, 10, new Random(1));

        System.out.println("\n***** J48 Algorithm Evaluation ***");
        System.out.println(evalJ48.toSummaryString("\n", true));
        System.out.println("Accuracy: " + evalJ48.pctCorrect() + "%");
        System.out.println("\nPrecision, Recall, F-Measure for each class:");
        System.out.println(evalJ48.toClassDetailsString());
        System.out.println("\nConfusion Matrix:");
        System.out.println(evalJ48.toMatrixString());
        
    }
    
    public static void trainingModelTime(Classifier classifier, Instances data) {
        long startTime = System.currentTimeMillis(); 
        try {
            classifier.buildClassifier(data);  
        } catch (Exception e) {
            e.printStackTrace();
        }
        long endTime = System.currentTimeMillis();  

        System.out.println("\nTakes " + (endTime - startTime) + " milliseconds to build the model!");
        System.out.println("____________________________________________________");
    }
}
