package model_evaluation;

import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import java.util.Random;

public class ModelEvaluator {

    public static void evaluateNaiveBayes(Instances train_data) throws Exception {
        Evaluation evalNB = new Evaluation(train_data);
        evalNB.crossValidateModel(new weka.classifiers.bayes.NaiveBayes(), train_data, 10, new Random(1));

        System.out.println("\n***** Naive Bayes Evaluation ***");
        System.out.println(evalNB.toSummaryString("\n", true));
        System.out.println("Accuracy: " + evalNB.pctCorrect() + "%");
        System.out.println("\nPrecision, Recall, F-Measure for each class:");
        System.out.println(evalNB.toClassDetailsString());
        System.out.println("\nConfusion Matrix:");
        System.out.println(evalNB.toMatrixString());
    }

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
}
