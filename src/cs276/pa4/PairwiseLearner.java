package cs276.pa4;

import java.util.*;

import weka.classifiers.Classifier;
import weka.classifiers.functions.LibSVM;
import weka.core.*;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Standardize;

/**
 * Implements Pairwise learner that can be used to train SVM
 */
public class PairwiseLearner extends Learner {
    private LibSVM model;

    public PairwiseLearner(boolean isLinearKernel) {
        try {
            model = new LibSVM();
        } catch (Exception e) {
            e.printStackTrace();
        }

        if (isLinearKernel) {
            model.setKernelType(new SelectedTag(LibSVM.KERNELTYPE_LINEAR, LibSVM.TAGS_KERNELTYPE));
        }
    }

    public PairwiseLearner(double C, double gamma, boolean isLinearKernel) {
        try {
            model = new LibSVM();
        } catch (Exception e) {
            e.printStackTrace();
        }

        model.setCost(C);
        model.setGamma(gamma); // only matter for RBF kernel
        if (isLinearKernel) {
            model.setKernelType(new SelectedTag(LibSVM.KERNELTYPE_LINEAR, LibSVM.TAGS_KERNELTYPE));
        }
    }

    @Override
    public Instances extractTrainFeatures(String train_data_file,
                                          String train_rel_file, Map<String, Double> idfs) {
        Map<Query, List<Document>> queryDocData = null;
        Map<String, Map<String, Double>> queryDocRel = null;
        try {
            queryDocData = Util.loadTrainData(train_data_file);
            queryDocRel = Util.loadRelData(train_rel_file);

        } catch (Exception e) {
            e.printStackTrace();
        }
        Instances dataset = null;

    /* Build attributes list */
        ArrayList<Attribute> attributes = new ArrayList<Attribute>();
        attributes.add(new Attribute("url_w"));
        attributes.add(new Attribute("title_w"));
        attributes.add(new Attribute("body_w"));
        attributes.add(new Attribute("header_w"));
        attributes.add(new Attribute("anchor_w"));
        attributes.add(new Attribute("relevance_score"));
        dataset = new Instances("train_dataset", attributes, 0);

    /* Add data */
        Feature feature = new Feature(idfs);
        int index = 0;
        Map<Query, List<Integer>> queryInstanceListMap = new HashMap<>();
        for (Query q : queryDocData.keySet()) {
            List<Integer> instanceList = new ArrayList<>();
            for (Document d : queryDocData.get(q)) {
                double[] feat = feature.extractFeatureVector(d, q);
                instanceList.add(index);
                dataset.add(new DenseInstance(1.0,
                        feature.addPreictedVarToFeatureVec(feat,
                                queryDocRel.get(q.query).get(d.url))));
                index++;
            }

            queryInstanceListMap.put(q, instanceList);
        }
        //Standardize the data
        Standardize filter = new Standardize();
        try {
            filter.setInputFormat(dataset);
            dataset = Filter.useFilter(dataset, filter);
        } catch (Exception e) {
            e.printStackTrace();
        }
        Instances diff_dataset=new Instances("diff_dataset",attributes,0);
        //building document pair indexes
        Map<Query, Map<Pair<Integer,Integer>, Integer>> pairwise_map=new HashMap<>();

        index=0;
        for (Query q: queryInstanceListMap.keySet()){
            Map<Pair<Integer, Integer>, Integer> dMap = new HashMap<>();
            List<Integer> instance_indx=queryInstanceListMap.get(q);
            int indx_size=instance_indx.size();
            for(int i=0;i<indx_size-1;i++){
                for (int j=i+1;j<indx_size;j++){
                    diff_dataset.add(new DenseInstance(1.0, instanceDiff(dataset.get(instance_indx.get(i)),
                            dataset.get(instance_indx.get(j)))));
                    Pair<Integer, Integer> doc_Pair = new Pair(i,j);
                    dMap.put(doc_Pair,index);
                    index++;
                }
            }
            pairwise_map.put(q,dMap);
        }
        diff_dataset=addClassLabel(diff_dataset,"train",attributes,"relevance");
        //diff_dataset.setClassIndex(diff_dataset.numAttributes()-1);
        Instances sample=balanceTestCases(diff_dataset);
        sample.setClassIndex(sample.numAttributes()-1);
        return sample;
    }

    @Override
    public Classifier training(Instances dataset) {
        try {
            model.buildClassifier(dataset);
        } catch (Exception e) {
            e.printStackTrace();
        }
        return model;
    }

    @Override
    public TestFeatures extractTestFeatures(String test_data_file,
                                            Map<String, Double> idfs) {
        try {
            Map<Query, List<Document>> testData = Util.loadTrainData(test_data_file);
            Instances dataset = null;
      /* Build attributes list */
            ArrayList<Attribute> attributes = new ArrayList<Attribute>();
            attributes.add(new Attribute("url_w"));
            attributes.add(new Attribute("title_w"));
            attributes.add(new Attribute("body_w"));
            attributes.add(new Attribute("header_w"));
            attributes.add(new Attribute("anchor_w"));
            attributes.add(new Attribute("relevance_score"));
            dataset = new Instances("test_dataset", attributes, 0);

            Map<Query, Map<Pair<Integer, Integer>, Integer>> pairwise_index_map = new HashMap<>();

            Feature feature = new Feature(idfs);
            int index = 0;
            Map<Query, List<Integer>> queryInstanceListMap = new HashMap<>();
            for (Query q : testData.keySet()) {
                List<Integer> instanceList = new ArrayList<>();
                for (Document d : testData.get(q)) {
                    double[] feat = feature.extractFeatureVector(d, q);
                    instanceList.add(index);
                    dataset.add(new DenseInstance(1.0,
                            feature.addPreictedVarToFeatureVec(feat,
                                    0.0D)));
                    index++;
                }

                queryInstanceListMap.put(q, instanceList);
            }

            //Standardize the data
            Standardize filter = new Standardize();
            filter.setInputFormat(dataset);
            dataset = Filter.useFilter(dataset, filter);

            Instances diff_dataset=new Instances("diff_dataset",attributes,0);
            //building document pair indexes
            Map<Query, Map<Pair<Integer,Integer>, Integer>> pairwise_map=new HashMap<>();

            index=0;
            for (Query q: queryInstanceListMap.keySet()){
                Map<Pair<Integer, Integer>, Integer> dMap = new HashMap<>();
                List<Integer> instance_indx=queryInstanceListMap.get(q);
                int indx_size=instance_indx.size();
                for(int i=0;i<indx_size-1;i++){
                    for (int j=i+1;j<indx_size;j++){
                        diff_dataset.add(new DenseInstance(1.0, instanceDiff(dataset.get(instance_indx.get(i)),
                                dataset.get(instance_indx.get(j)))));
                        Pair<Integer, Integer> doc_Pair = new Pair(i,j);
                        dMap.put(doc_Pair,index);
                        index++;
                    }
                }
                pairwise_map.put(q,dMap);
            }
            diff_dataset=addClassLabel(diff_dataset,"test",attributes,"relevance");
            diff_dataset.setClassIndex(diff_dataset.numAttributes()-1);

            TestFeatures tf = new TestFeatures();
            tf.features = diff_dataset;
            tf.pairwise_index_map = pairwise_map;
            return tf;

        } catch (Exception e) {
            e.printStackTrace();
        }
        return null;
    }

    @Override
    public Map<Query, List<Document>> testing(TestFeatures tf,
                                              Classifier model) {
        // read the test data...
        Instances testInstance = tf.features;
        Map<Query, List<Document>> rankings = new HashMap<>();
        Map<Query, Map<Pair<Document, Document>, Integer>> pairwise_map = tf.pairwise_index_map;

        Map<Document, Integer> documentCountMap = new TreeMap<>();

        for (Query q : pairwise_map.keySet()) {

            Map<Pair<Document, Document>, Integer> ind_map = pairwise_map.get(q);

            for (Pair<Document, Document> documentPair : ind_map.keySet()) {
                double prediction = Double.MIN_VALUE;
                Integer index = ind_map.get(documentPair);
                try {
                    prediction = model.classifyInstance(testInstance.get(index));
                    if (prediction == 0) {

                        incrementOrInit(documentCountMap, documentPair.getFirst());
//                        documentCountMap.put(documentPair.getFirst(), )
                    } else {
                        incrementOrInit(documentCountMap, documentPair.getSecond());

                    }

                } catch (Exception e) {
                    System.err.println("Error classifying " + documentPair.getFirst().url + " and " + documentPair.getSecond());
                }

            }

//            List<Pair<Document, Double>> list = new ArrayList<>();
//            for (Document d : documentMap.keySet()) {
//                double prediction = Double.MIN_VALUE;
//                Integer index = documentMap.get(d);
//                try {
//                    prediction = model.classifyInstance(testInstance.get(index));
//                } catch (Exception e) {
//                    System.err.println("Error classifying " + d.url);
//                }
//                Pair<Document, Double> p = new Pair<>(d, prediction);
//                list.add(p);
//            }
//            list.sort(new Comparator<Pair<Document, Double>>() {
//                @Override
//                public int compare(Pair<Document, Double> o1, Pair<Document, Double> o2) {
//                    return o2.getSecond().compareTo(o1.getSecond());
//                }
//            });
//            List<Document> documentList = new ArrayList<>();
//            for (Pair<Document, Double> pair : list) {
//                documentList.add(pair.getFirst());
//            }
            //sort the map by value..
            Map<Document, Integer> sortedDocuments = sortByValue(documentCountMap);
            //add all the keys (Documents) to list...
            ArrayList<Document> documentList = new ArrayList<>(sortedDocuments.keySet());
//            sortedDocuments.

//            for (Document document: documentCountMap.keySet()) {
//                documentList.add(document);
//            }
            rankings.put(q, documentList);
        }

        return rankings;
    }


    private static <K, V extends Comparable<? super V>> Map<K, V>
    sortByValue( Map<K, V> map )
    {
        List<Map.Entry<K, V>> list =
                new LinkedList<>( map.entrySet() );
        Collections.sort( list, new Comparator<Map.Entry<K, V>>()
        {
            @Override
            public int compare( Map.Entry<K, V> o1, Map.Entry<K, V> o2 )
            {
                return ( o1.getValue() ).compareTo( o2.getValue() );
            }
        } );

        Map<K, V> result = new LinkedHashMap<>();
        for (Map.Entry<K, V> entry : list)
        {
            result.put( entry.getKey(), entry.getValue() );
        }
        return result;
    }

    static <K,V extends Comparable<? super V>>
    SortedSet<Map.Entry<K,V>> entriesSortedByValues(Map<K,V> map) {
        SortedSet<Map.Entry<K,V>> sortedEntries = new TreeSet<Map.Entry<K,V>>(
                new Comparator<Map.Entry<K,V>>() {
                    @Override public int compare(Map.Entry<K,V> e1, Map.Entry<K,V> e2) {
                        int res = e1.getValue().compareTo(e2.getValue());
                        return res != 0 ? res : 1;
                    }
                }
        );
        sortedEntries.addAll(map.entrySet());
        return sortedEntries;
    }

    private Instances documentPairInstances(Instances instances,
                                            String name, ArrayList<Attribute> attributes,
                                            String newAttName, Map<Query, List<Integer>> queryInstanceListMap) {

        Instances dataset = new Instances(name, attributes, 0);
        for (Query query : queryInstanceListMap.keySet()) {
            for (int i=0;i<queryInstanceListMap.get(query).size()-1;i++) {
                for (int j=i+1;j<queryInstanceListMap.get(query).size();j++) {
                    dataset.add(new DenseInstance(1.0, instanceDiff(instances.get(i),
                            instances.get(j))));
                }
            }
        }
        dataset = addClassLabel(dataset, name, attributes, newAttName);
        return dataset;
    }

    private void incrementOrInit(Map<Document, Integer> documentCountMap, Document key) {

        if (documentCountMap.get(key)==null) {
            documentCountMap.put(key, 1);
        } else {
            documentCountMap.put(key, documentCountMap.get(key) + 1);
        }

    }

    private double[] instanceDiff(Instance inst1, Instance inst2) {
        double[] arr1 = inst1.toDoubleArray();
        double[] arr2 = inst2.toDoubleArray();
        double[] res = new double[arr1.length];

        //assuming the last column is the class index
        for (int i = 0; i < arr1.length; i++)
            res[i] = arr1[i] - arr2[i];

        return res;
    }

    private Instances addClassLabel(Instances instances,
                                    String name, ArrayList<Attribute> attributes,
                                    String newAttributeName) {

        ArrayList<String> labels = new ArrayList<String>();
        labels.add("+1");
        labels.add("-1");
        Attribute relevance = new Attribute(newAttributeName, labels);
        attributes.add(relevance);

        Instances dataset = new Instances(name, attributes, 0);

        for (int i = 0; i < instances.size(); i++) {
            double[] val = new double[attributes.size()];
            double[] rec = instances.get(i).toDoubleArray();
            for (int j = 0; j < rec.length; j++) {
                val[j] = rec[j];
            }
            if (rec[rec.length - 1] >= 0)
                val[val.length - 1] = relevance.indexOfValue("+1");
            else
                val[val.length - 1] = relevance.indexOfValue("-1");
            dataset.add(new DenseInstance(1.0, val));
        }

        dataset.deleteAttributeAt(dataset.numAttributes() - 2);
        /*
        CSVSaver saver = new CSVSaver();
        try {
            if (name.equals("test_dataset"))
                saver.setFile(new File("test.csv"));
            else
                saver.setFile(new File("train.csv"));
          saver.setInstances(dataset);
          saver.writeBatch();
        } catch (IOException e) {
          e.printStackTrace();
        }
        */
        return dataset;
    }

    private Instances balanceTestCases(Instances instances){
        ArrayList<Attribute> attribs=new ArrayList<>();
        Enumeration<Attribute> enumeration=instances.enumerateAttributes();
        while(enumeration.hasMoreElements())
            attribs.add(enumeration.nextElement());

        Instances positive = new Instances("positive",attribs,0);
        Instances negative = new Instances("negative",attribs,0);
        Instances result = new Instances("balance",attribs,0);

        for(Instance i:instances){
            double[] vals=i.toDoubleArray();
            if (vals[vals.length-1]==0D) //index for +1
                positive.add(i);
            else
                negative.add(i);
        }
        int eachGroup=negative.size();
        boolean negGroup=true;

        if (positive.size()<negative.size()) {
            eachGroup = positive.size();
            negGroup=false;
        }

        if(negGroup){
            result=negative;
            Random random=new Random(1000);
            for(int k=0;k<eachGroup;k++){
                int r=random.nextInt(positive.size());
                result.add(positive.instance(r));
            }
        }else{
            result=positive;
            Random random=new Random(1000);
            for(int k=0;k<eachGroup;k++){
                int r=random.nextInt(negative.size());
                result.add(negative.instance(r));
            }
        }
        return result;

    }

}
