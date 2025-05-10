package model_training;

import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.ArffLoader;
import weka.core.converters.ArffSaver;
import java.io.File;
import java.io.IOException;

public class ModelTrainer {

    public static Instances trainJ48(String filePath) throws Exception {
        ArffLoader loader = new ArffLoader();
        loader.setSource(new File(filePath));
        Instances train_data = loader.getDataSet();
        train_data.setClassIndex(train_data.numAttributes() - 1); 

        J48 tree = new J48();
        String[] options = {"-C", "0.1", "-M", "2"};
        tree.setOptions(options);
        tree.buildClassifier(train_data);

        Instances predictionsJ48 = new Instances(train_data);
        for (int i = 0; i < train_data.numInstances(); i++) {
            double predictedClass = tree.classifyInstance(train_data.instance(i)); 
            predictionsJ48.instance(i).setClassValue(predictedClass);  
        }

        // Lưu kết quả dự đoán vào file ARFF
        ArffSaver saverJ48 = new ArffSaver();
        saverJ48.setInstances(predictionsJ48);  
        saverJ48.setFile(new File("C:\\Users\\admin\\eclipse-workspace\\DataMining-FinalProject\\src\\Data//j48_predictions.arff"));
        try {
            saverJ48.writeBatch();
        } catch (IOException e) {
            System.out.println("Lỗi khi lưu file ARFF J48: " + e.getMessage());
            e.printStackTrace();
        }

        return train_data; 
    }

     //cải thiện với Naive Bayes
    public static Instances trainJ48WithNaiveBayesImprovement(String filePath) throws Exception {
        ArffLoader loader = new ArffLoader();
        loader.setSource(new File(filePath));
        Instances train_data = loader.getDataSet();
        train_data.setClassIndex(train_data.numAttributes() - 1); // Thiết lập chỉ mục lớp (class)

        // 1. Huấn luyện mô hình J48
        J48 tree = new J48();
        tree.buildClassifier(train_data);  // Xây dựng mô hình J48 trên dữ liệu gốc

        // 2. Dự đoán lớp cho dữ liệu bằng J48
        Instances predictionsJ48 = new Instances(train_data);
        for (int i = 0; i < train_data.numInstances(); i++) {
            double predictedClass = tree.classifyInstance(train_data.instance(i)); 
            predictionsJ48.instance(i).setClassValue(predictedClass);  
        }

        // 3. Sử dụng Naive Bayes để học từ các kết quả dự đoán của J48
        NaiveBayes nb = new NaiveBayes();
        nb.buildClassifier(predictionsJ48); // Huấn luyện Naive Bayes với dự đoán của J48

        // 4. Dự đoán lớp mới với Naive Bayes (cải thiện)
        Instances improvedPredictions = new Instances(train_data);
        for (int i = 0; i < train_data.numInstances(); i++) {
            double predictedClassNB = nb.classifyInstance(predictionsJ48.instance(i)); 
            improvedPredictions.instance(i).setClassValue(predictedClassNB);  // Dự đoán mới từ Naive Bayes
        }

        // Lưu kết quả dự đoán cải thiện vào file ARFF
        ArffSaver saverImproved = new ArffSaver();
        saverImproved.setInstances(improvedPredictions);  
        saverImproved.setFile(new File("C:\\Users\\admin\\eclipse-workspace\\DataMining-FinalProject\\src\\Data//j48_with_naivebayes_predictions.arff"));
        try {
            saverImproved.writeBatch();
        } catch (IOException e) {
            System.out.println("Error when save file ARFF Naive Bayes: " + e.getMessage());
            e.printStackTrace();
        }

        return improvedPredictions; 
    }
}