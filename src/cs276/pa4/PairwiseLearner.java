package cs276.pa4;

import java.io.File;
import java.io.IOException;
import java.util.*;

import weka.classifiers.Classifier;
import weka.classifiers.functions.LibSVM;
import weka.classifiers.functions.LinearRegression;
import weka.core.*;
import weka.core.converters.CSVSaver;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Standardize;

/**
 * Implements Pairwise learner that can be used to train SVM
 *
 */
public class PairwiseLearner extends Learner {
  private LibSVM model;
  public PairwiseLearner(boolean isLinearKernel){
    try{
      model = new LibSVM();
    } catch (Exception e){
      e.printStackTrace();
    }

    if(isLinearKernel){
      model.setKernelType(new SelectedTag(LibSVM.KERNELTYPE_LINEAR, LibSVM.TAGS_KERNELTYPE));
    }
  }

  public PairwiseLearner(double C, double gamma, boolean isLinearKernel){
    try{
      model = new LibSVM();
    } catch (Exception e){
      e.printStackTrace();
    }

    model.setCost(C);
    model.setGamma(gamma); // only matter for RBF kernel
    if(isLinearKernel){
      model.setKernelType(new SelectedTag(LibSVM.KERNELTYPE_LINEAR, LibSVM.TAGS_KERNELTYPE));
    }
  }

  @Override
  public Instances extractTrainFeatures(String train_data_file,
                                        String train_rel_file, Map<String, Double> idfs) {
    Map<Query,List<Document>> queryDocData=null;
    Map<String, Map<String, Double>> queryDocRel=null;
    try {
      queryDocData=Util.loadTrainData(train_data_file);
      queryDocRel=Util.loadRelData(train_rel_file);

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
    Feature feature=new Feature(idfs);

    for(Query q:queryDocData.keySet()){
      for (Document d:queryDocData.get(q)){

        double[] feat=feature.extractFeatureVector(d,q);
        dataset.add(new DenseInstance(1.0,
                feature.addPreictedVarToFeatureVec(feat,
                queryDocRel.get(q.query).get(d.url))));
      }
    }

    //Standardize the data
    Standardize filter = new Standardize();
    try {
      filter.setInputFormat(dataset);
      dataset=Filter.useFilter(dataset, filter);
    } catch (Exception e) {
      e.printStackTrace();
    }
    dataset=documentPairInstances(dataset,"train_dataset",attributes,"relevance");

    /* Set last attribute as target */
    dataset.setClassIndex(dataset.numAttributes() - 1);
    return dataset;
  }

  @Override
  public Classifier training(Instances dataset) {
    Classifier classifier = new LibSVM();
    try {
      classifier.buildClassifier(dataset);
    } catch (Exception e) {
      e.printStackTrace();
    }
    return classifier;
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
      Map<Query,Map<Document,Integer>> indexMap=new HashMap<>();
      int index=0;
      Feature feature = new Feature(idfs);
      for (Query q:testData.keySet()){
        Map<Document, Integer> docMap = new HashMap<>();
        for(Document d:testData.get(q)){
          /*
          double[] instance=Util.getInstance(q,d,idfs,null);
          dataset.add(new DenseInstance(1.0,instance));
          */
          double[] feat=feature.extractFeatureVector(d,q);
          dataset.add(new DenseInstance(1.0,feature.addPreictedVarToFeatureVec(feat,0.0)));
          docMap.put(d, index);
          ++index;
        }
        indexMap.put(q,docMap);
      }

      //Standardize the data
      Standardize filter = new Standardize();
      filter.setInputFormat(dataset);
      dataset=Filter.useFilter(dataset, filter);

      dataset=documentPairInstances(dataset,"test_dataset",attributes,null);
      TestFeatures tf = new TestFeatures();
      tf.features=dataset;
      tf.index_map=indexMap;
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
    Map<Query, Map<Document , Integer>> indexMap =  tf.index_map;

    for (Query q: indexMap.keySet()) {

      Map<Document, Integer> documentMap = indexMap.get(q);
      List<Pair<Document,Double>> list = new ArrayList<>();
      for (Document d: documentMap.keySet()){
        double prediction = Double.MIN_VALUE;
        Integer index = documentMap.get(d);
        try{
          prediction = model.classifyInstance(testInstance.get(index));
        }catch(Exception e){
          System.err.println("Error classifying " + d.url);
        }
        Pair<Document,Double> p = new Pair<>(d, prediction);
        list.add(p);
      }
      list.sort(new Comparator<Pair<Document, Double>>() {
        @Override
        public int compare(Pair<Document, Double> o1, Pair<Document, Double> o2) {
          return o2.getSecond().compareTo(o1.getSecond());
        }
      });
      List<Document> documentList = new ArrayList<>();
      for (Pair<Document, Double> pair: list) {
        documentList.add(pair.getFirst());
      }
      rankings.put(q, documentList);
    }

    return rankings;
  }

  private Instances documentPairInstances(Instances instances,
                                              String name, ArrayList<Attribute> attributes,
                                          String newAttName){
    Instances dataset=new Instances(name,attributes,0);
    for(int i=0;i<instances.size()-1;i++) {
      for (int j = i + 1; j < instances.size(); j++) {
        dataset.add(new DenseInstance(1.0,instanceDiff(instances.get(i),
                instances.get(j))));
      }
    }
    //changing the class variable to Categorical
    if (newAttName!=null)
      dataset=addClassLabel(dataset,name,attributes,newAttName);
    return dataset;
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
                                             String newAttributeName){

    ArrayList<String> labels = new ArrayList<String>();
    labels.add("+1");
    labels.add("-1");
    Attribute relevance = new Attribute(newAttributeName, labels);
    attributes.add(relevance);

    Instances dataset=new Instances(name,attributes,0);

    for(int i=0;i<instances.size();i++){
      double[] val=new double[attributes.size()];
      double[] rec=instances.get(i).toDoubleArray();
      for(int j=0;j<rec.length;j++){
        val[j]=rec[j];
      }
      if (rec[rec.length-1]>=0)
        val[val.length-1]=relevance.indexOfValue("+1");
      else
        val[val.length-1]=relevance.indexOfValue("-1");
      dataset.add(new DenseInstance(1.0,val));
    }
    /*
    CSVSaver saver = new CSVSaver();
    try {
      saver.setFile(new File("test.csv"));
      saver.setInstances(dataset);
      saver.writeBatch();
    } catch (IOException e) {
      e.printStackTrace();
    }
    */
    return dataset;
  }

}
