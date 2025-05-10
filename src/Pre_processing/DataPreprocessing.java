package Pre_processing;

import weka.core.Attribute;
import weka.core.Instances;

import weka.core.converters.CSVLoader;
import weka.core.converters.ArffSaver;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.StringToNominal;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;
//import weka.filters.unsupervised.attribute.Discretize;
import weka.filters.unsupervised.attribute.Remove;

import java.io.File;
import java.util.ArrayList;

import java.util.HashSet;
import java.util.Set;


public class DataPreprocessing {

    public static Instances loadData(String filePath) throws Exception {
        CSVLoader loader = new CSVLoader();
        loader.setSource(new File(filePath));
        return loader.getDataSet();
    }
    
    
    public static Instances convertByteStringsToString(Instances data) {
        for (int i = 0; i < data.numInstances(); i++) {
            for (int j = 0; j < data.numAttributes(); j++) {
                if (data.attribute(j).isNominal()) {
                    String value = data.instance(i).stringValue(j);
                    // Chuyển đổi giá trị byte thành string nếu có
                    if (value.startsWith("b'")) {
                        value = value.substring(2, value.length() - 1);  // Loại bỏ 'b' và dấu nháy
                        data.instance(i).setValue(j, value);
                    }
                }
            }
        }
        return data;
    }


 
    public static Instances removeAttributes(Instances data, String... columns) {
        Instances instances = null;
        try {
            Remove remove = new Remove();
            
            StringBuilder columnIndices = new StringBuilder();
            for (int i = 0; i < columns.length; i++) {
                columnIndices.append(columns[i]);
                if (i < columns.length - 1) {
                    columnIndices.append(",");
                }
            }
            remove.setAttributeIndices(columnIndices.toString());  // Chỉ định các cột cần loại bỏ
            remove.setInputFormat(data);
            instances = Filter.useFilter(data, remove);
            System.out.println("Successfully removed attributes: " + columnIndices.toString());

        } catch (Exception e) {
            System.err.println("Error in removing attribute");
            System.err.println(e);
        }
        return instances;
    }


    // Phương thức xử lý giá trị thiếu
    public static Instances replaceMissingValues(Instances data) throws Exception {
        Instances instances = null;
        try {
            ReplaceMissingValues replaceMissingValues = new ReplaceMissingValues();
            replaceMissingValues.setInputFormat(data);
            instances = Filter.useFilter(data, replaceMissingValues);
            System.out.println("Missing values replaced successfully!");
        } catch (Exception e) {
            System.err.println("Error in handling missing values!");
            System.err.println(e);
        }
        return instances;
    }
    
    public static Instances removeOutliers(Instances data, String... attributeNames) {
        Instances instances = new Instances(data);

        for (String attributeName : attributeNames) {
            Attribute attribute = data.attribute(attributeName);
            if (attribute == null) {
                System.out.println("Column doesn't exist: " + attributeName);
                continue; 
            }

            double q1 = calculatePercentile(instances, attribute, 25);
            double q3 = calculatePercentile(instances, attribute, 75);
            double iqr = q3 - q1;

            double lowerBound = q1 - 1.5 * iqr;
            double upperBound = q3 + 1.5 * iqr;

            Instances dataAfterFiltering = new Instances(instances);
            for (int i = 0; i < dataAfterFiltering.numInstances(); i++) {
                double value = dataAfterFiltering.instance(i).value(attribute);
                if (value < lowerBound || value > upperBound) {
                    dataAfterFiltering.delete(i); 
                }
            }
            instances = dataAfterFiltering;
        }

        System.out.println("Remove outliers successfully");
        return instances;
    }
    
 // Phương thức tính toán percentiles (Q1, Q3)
    private static double calculatePercentile(Instances data, Attribute attribute, double percentile) {
        double[] values = new double[data.numInstances()];
        for (int i = 0; i < data.numInstances(); i++) {
            values[i] = data.instance(i).value(attribute);
        }
        java.util.Arrays.sort(values);

        int index = (int) Math.ceil(percentile / 100.0 * values.length) - 1;
        return values[index];
    }
    
    public static Instances removeOutliersByAttributeName(Instances data, String attributeName, String... validValues) {
//		Instances instances = new Instances(data);  

        Attribute attribute = data.attribute(attributeName);
        if (attribute == null) {
            System.out.println("Column doesn't exist: " + attributeName);
            return data;  
        }
        Set<String> validValueSet = new HashSet<>();
        for (String value : validValues) {
            validValueSet.add(value.trim().toLowerCase());  // Chuẩn hóa chữ hoa, chữ thường và loại bỏ khoảng trắng
        }

        ArrayList<Integer> validIndices = new ArrayList<>();

        for (int i = 0; i < data.numInstances(); i++) {
            String value = data.instance(i).stringValue(attribute).trim().toLowerCase();  // Chuẩn hóa giá trị
            // Nếu giá trị hợp lệ, lưu chỉ số
            if (validValueSet.contains(value)) {
                validIndices.add(i);
            }
        }

        // Tạo một Instances mới chỉ chứa các giá trị hợp lệ
        Instances filteredData = new Instances(data, validIndices.size());
        for (Integer index : validIndices) {
            filteredData.add(data.instance(index));
        }

        System.out.println("Removed outliers from attribute: " + attributeName);
        return filteredData;  // Trả về dataset đã xử lý
    }

    
    
//    @SuppressWarnings("serial")
//	public static Instances labelEncoding(Instances data, int attributeNumber, Object... keyAndValuePairs) {
//        Instances instances = null;
//        HashMap<String, Integer> encoder = new HashMap<>();
//
//        // Kiểm tra nếu dữ liệu trống hoặc cột không tồn tại
//        if (data == null || attributeNumber < 0 || attributeNumber >= data.numAttributes()) {
//            System.out.println("Error: Data or column index is invalid.");
//            return data;
//        }
//
//        // Tạo bảng mã hóa từ các giá trị phân loại và giá trị số
//        try {
//            // Tạo bảng mã hóa cho các giá trị phân loại
//            for (int i = 0; i < keyAndValuePairs.length; i += 2) {
//                encoder.put((String) keyAndValuePairs[i], (Integer) keyAndValuePairs[i + 1]);
//            }
//
//            Attribute attribute = data.attribute(attributeNumber);
//            String attributeName = attribute.name();
//            Attribute encodedAttribute = new Attribute(attributeName + "_encoded");
//
//            // Tạo đối tượng Instances mới với một cột cho giá trị đã mã hóa
//            Instances instances2 = new Instances("EncodedData", new ArrayList<Attribute>() {{
//                add(encodedAttribute);
//            }}, data.numInstances());
//
//            // Duyệt qua tất cả các instance và thay thế giá trị phân loại bằng số
//            for (int i = 0; i < data.numInstances(); i++) {
//                Instance instance = data.instance(i);
//                String categoryValue = instance.stringValue(attribute);
//                Integer encodedValue = encoder.get(categoryValue);
//
//                if (encodedValue != null) {
//                    // Thêm giá trị đã mã hóa vào instances mới
//                    DenseInstance newInstance = new DenseInstance(1);
//                    newInstance.setValue(encodedAttribute, encodedValue);
//                    instances2.add(newInstance);
//                } else {
//                    System.out.println("Warning: Value not found for encoding in column " + attributeName);
//                }
//            }
//
//            // Merge instances đã mã hóa với dataset gốc
//            instances = Instances.mergeInstances(data, instances2);
//            instances.deleteAttributeAt(attributeNumber);  // Loại bỏ cột gốc đã mã hóa
//
//            System.out.println("Label encoding successfully!");
//        } catch (Exception e) {
//            System.out.println("Error in encoding!");
//            e.printStackTrace();
//        }
//
//        return instances;
//    }

    
    // Phương thức chuyển các cột số thành cột phân loại (Numeric to Nominal)
    public static Instances convertNumericToNominal(Instances data) throws Exception {
        StringToNominal numericToNominal = new StringToNominal();
        numericToNominal.setAttributeRange("first-last");
        numericToNominal.setInputFormat(data);
        return Filter.useFilter(data, numericToNominal);
    }

//    // Phương thức Discretizing: Chuyển đổi các thuộc tính liên tục thành phân loại
//    public static Instances discretize(Instances data) throws Exception {
//        Discretize discretize = new Discretize();
//        discretize.setInputFormat(data);
//        return Filter.useFilter(data, discretize);
//    }


    public static void saveDataToArff(Instances data, String filePath) throws Exception {
        ArffSaver saver = new ArffSaver();
        saver.setInstances(data);
        saver.setFile(new File(filePath));
        saver.writeBatch();
    }

    
    // Phương thức xử lý tiền xử lý chính
    public static void preprocess() throws Exception {
        Instances data = loadData("C:\\Users\\admin\\eclipse-workspace\\DataMining-FinalProject\\src\\Data\\survey.csv");
        data = convertByteStringsToString(data);
        data = removeAttributes(data, "1", "5", "27");
        data = replaceMissingValues(data);
        data = removeOutliers(data, "Age","", "Income", "Height");
        data = removeOutliersByAttributeName(data, "Gender", "Male", "Female", "Non-binary", "Cis Male", "Cis Female", "Trans Female", "Agender");


//        data = labelEncoding(data, data.attribute("Gender").index(), "Male", 1, "Female", 2, "Other", 3);
//
//        // 2. Label Encoding cho cột "Country"
//        data = labelEncoding(data, data.attribute("Country").index(), 
//                             "United States", 1, "Canada", 2, "United Kingdom", 3, 
//                             "Bulgaria", 4, "France", 5, "Portugal", 6);
//
//        // 3. Label Encoding cho cột "Leave"
//        data = labelEncoding(data, data.attribute("Leave").index(), 
//                             "Somewhat easy", 1, "Somewhat difficult", 2, "Very easy", 3, 
//                             "Very difficult", 4, "Don't know", 5);
//
//        // 4. Label Encoding cho cột "Mental_health_consequence"
//        data = labelEncoding(data, data.attribute("Mental_health_consequence").index(),
//                             "No", 1, "Maybe", 2, "Yes", 3);
//
//        // 5. Label Encoding cho cột "Phys_health_consequence"
//        data = labelEncoding(data, data.attribute("Phys_health_consequence").index(),
//                             "No", 1, "Yes", 2, "Maybe", 3);
//
//        // 6. Label Encoding cho cột "Coworkers"
//        data = labelEncoding(data, data.attribute("Coworkers").index(),
//                             "Some of them", 1, "No", 2, "Yes", 3);
//
//        // 7. Label Encoding cho cột "Supervisor"
//        data = labelEncoding(data, data.attribute("Supervisor").index(),
//                             "Yes", 1, "No", 2, "Some of them", 3);
//
//        // 8. Label Encoding cho cột "Mental_health_interview"
//        data = labelEncoding(data, data.attribute("Mental_health_interview").index(),
//                             "No", 1, "Yes", 2, "Maybe", 3);
//
//        // 9. Label Encoding cho cột "Phys_health_interview"
//        data = labelEncoding(data, data.attribute("Phys_health_interview").index(),
//                             "Maybe", 1, "No", 2, "Yes", 3);
//
//        // 10. Label Encoding cho cột "Mental_vs_physical"
//        data = labelEncoding(data, data.attribute("Mental_vs_physical").index(),
//                             "Yes", 1, "Don't know", 2, "No", 3);
//
//        // 11. Label Encoding cho cột "Obs_consequence"
//        data = labelEncoding(data, data.attribute("Obs_consequence").index(),
//                             "No", 1, "Yes", 2);
//
//        data = convertNumericToNominal(data);
//        data = discretize(data);
        saveDataToArff(data, "C:\\Users\\admin\\eclipse-workspace\\DataMining-FinalProject\\src\\Data\\cleaned_data.arff");
    }
}