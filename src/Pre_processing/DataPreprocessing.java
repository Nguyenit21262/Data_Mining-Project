package Pre_processing;

import weka.core.Attribute;
import weka.core.Instances;

import weka.core.converters.CSVLoader;
import weka.core.converters.ArffSaver;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.StringToNominal;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;
import weka.filters.unsupervised.attribute.Remove;
import java.io.File;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.Set;


public class DataPreprocessing {

	public static Instances loadData(String filePath) {
        try {
            CSVLoader loader = new CSVLoader();
            loader.setSource(new File(filePath));
            System.out.println("Data loaded successfully");
            return loader.getDataSet();
        } catch (Exception e) {
            System.err.println("Error loading data from: " + filePath);
            e.printStackTrace();
            return null;
        }
    }

    public static Instances convertByteStringsToString(Instances data) {
        try {
            for (int i = 0; i < data.numInstances(); i++) {
                for (int j = 0; j < data.numAttributes(); j++) {
                    if (data.attribute(j).isNominal()) {
                        String value = data.instance(i).stringValue(j);
                        // Convert byte values to string
                        if (value.startsWith("b'")) {
                            value = value.substring(2, value.length() - 1); // Remove 'b' and quotes
                            data.instance(i).setValue(j, value);
                        }
                    }
                }
            }
            System.out.println("Byte strings converted to regular strings successfully!");
            return data;
        } catch (Exception e) {
            System.err.println("Error in converting byte strings to string.");
            e.printStackTrace();
            return null;
        }
    }

    public static Instances removeAttributes(Instances data, String... columns) {
        try {
            Remove remove = new Remove();
            StringBuilder columnIndices = new StringBuilder();
            for (int i = 0; i < columns.length; i++) {
                columnIndices.append(columns[i]);
                if (i < columns.length - 1) {
                    columnIndices.append(",");
                }
            }
            remove.setAttributeIndices(columnIndices.toString());  // Specify columns to remove
            remove.setInputFormat(data);
            Instances filteredData = Filter.useFilter(data, remove);
            System.out.println("Successfully removed attributes: " + columnIndices.toString());
            return filteredData;
        } catch (Exception e) {
            System.err.println("Error in removing attributes.");
            e.printStackTrace();
            return null;
        }
    }

    public static Instances replaceMissingValues(Instances data) {
        try {
            ReplaceMissingValues replaceMissingValues = new ReplaceMissingValues();
            replaceMissingValues.setInputFormat(data);
            Instances instances = Filter.useFilter(data, replaceMissingValues);
            System.out.println("Missing values replaced successfully!");
            return instances;
        } catch (Exception e) {
            System.err.println("Error in handling missing values!");
            e.printStackTrace();
            return null;
        }
    }

    public static Instances removeOutliers(Instances data, String... attributeNames) {
        try {
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
            System.out.println("Outliers removed successfully!");
            return instances;
        } catch (Exception e) {
            System.err.println("Error in removing outliers.");
            e.printStackTrace();
            return null;
        }
    }

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
        try {
            Attribute attribute = data.attribute(attributeName);
            if (attribute == null) {
                System.out.println("Column doesn't exist: " + attributeName);
                return data;
            }

            Set<String> validValueSet = new HashSet<>();
            for (String value : validValues) {
                validValueSet.add(value.trim().toLowerCase());  
            }

            ArrayList<Integer> validIndices = new ArrayList<>();
            for (int i = 0; i < data.numInstances(); i++) {
                String value = data.instance(i).stringValue(attribute).trim().toLowerCase();
                if (validValueSet.contains(value)) {
                    validIndices.add(i);
                }
            }

            // Create a new Instances object with valid values
            Instances filteredData = new Instances(data, validIndices.size());
            for (Integer index : validIndices) {
                filteredData.add(data.instance(index));
            }

            System.out.println("Outliers removed from attribute: " + attributeName);
            return filteredData;
        } catch (Exception e) {
            System.err.println("Error in removing outliers by attribute.");
            e.printStackTrace();
            return null;
        }
    }

    public static Instances convertNumericToNominal(Instances data) {
        try {
            StringToNominal numericToNominal = new StringToNominal();
            numericToNominal.setAttributeRange("first-last");
            numericToNominal.setInputFormat(data);
            Instances convertedData = Filter.useFilter(data, numericToNominal);
            System.out.println("Numeric attributes converted to nominal successfully!");
            return convertedData;
        } catch (Exception e) {
            System.err.println("Error in converting numeric to nominal.");
            e.printStackTrace();
            return null;
        }
    }

    public static void saveDataToArff(Instances data, String filePath) {
        try {
            ArffSaver saver = new ArffSaver();
            saver.setInstances(data);
            saver.setFile(new File(filePath));
            saver.writeBatch();
            System.out.println("Data saved to ARFF file successfully");
        } catch (Exception e) {
            System.err.println("Error saving data to ARFF file.");
            e.printStackTrace();
        }
    }

    
    public static void preprocess() throws Exception {
        Instances data = loadData("C:\\Users\\admin\\eclipse-workspace\\DataMining-FinalProject\\src\\Data\\survey.csv");
        data = convertByteStringsToString(data);
        data = removeAttributes(data, "1", "5", "27");
        data = replaceMissingValues(data);
        data = removeOutliers(data, "Age","", "Income", "Height");
//        data = removeOutliersByAttributeName(data, "Gender", "Male", "Female", "Non-binary", "Cis Male", "Cis Female", "Trans Female", "Agender");
        saveDataToArff(data, "C:\\Users\\admin\\eclipse-workspace\\DataMining-FinalProject\\src\\Data\\cleaned_data.arff");
    }
}