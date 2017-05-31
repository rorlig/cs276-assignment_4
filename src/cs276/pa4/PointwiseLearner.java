package cs276.pa4;

import java.util.*;

import weka.classifiers.Classifier;
import weka.classifiers.functions.LinearRegression;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Implements point-wise learner that can be used to implement logistic regression
 *
 */
public class PointwiseLearner extends Learner {

  @Override
  public Instances extractTrainFeatures(String train_data_file,
      String train_rel_file, Map<String, Double> idfs) {
    
    /*
     * @TODO: Below is a piece of sample code to show 
     * you the basic approach to construct a Instances 
     * object, replace with your implementation. 
     */
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

    for(Query q:queryDocData.keySet()){
      for (Document d:queryDocData.get(q)){
        double[] instance=Util.getInstance(q,d,idfs,queryDocRel);
        Instance inst = new DenseInstance(1.0, instance);
        dataset.add(inst);
      }
    }
    /* Set last attribute as target */
    dataset.setClassIndex(dataset.numAttributes() - 1);
    return dataset;
  }

  @Override
  public Classifier training(Instances dataset) {
    Classifier classifier = new LinearRegression();
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
      for (Query q:testData.keySet()){
        Map<Document, Integer> docMap = new HashMap<>();
        for(Document d:testData.get(q)){
          double[] instance=Util.getInstance(q,d,idfs,null);
          dataset.add(new DenseInstance(1.0,instance));
          docMap.put(d, index);
          ++index;
        }
        indexMap.put(q,docMap);
      }
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

}
