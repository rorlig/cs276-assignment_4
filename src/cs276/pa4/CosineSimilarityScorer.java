package cs276.pa4;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;

/**
 * Skeleton code for the implementation of a 
 * Cosine Similarity Scorer in Task 1.
 */
public class CosineSimilarityScorer extends AScorer {

  /*
   * TODO: You will want to tune the values for
   * the weights for each field.
   */
  double urlweight = 0.3;
  double titleweight  = 0.6;
  double bodyweight = 0.3;
  double headerweight = 0.8;
  double anchorweight = 0.5;
  double smoothingBodyLength = 1.0;
  
  /**
   * Construct a Cosine Similarity Scorer.
   * @param idfs the map of idf values
   */
  public CosineSimilarityScorer(Map<String,Double> idfs) {
    super(idfs);
  }

  /**
   * Get the net score for a query and a document.
   * @param tfs the term frequencies
   * @param q the Query
   * @param tfQuery the term frequencies for the query
   * @param d the Document
   * @return the net score
   */
  public double getNetScore(Map<String, Map<String, Double>> tfs, Query q, Map<String,Double> tfQuery, Document d) {
    double score = 0.0;
    Map<String,ArrayList<Double>> typeVec=new HashMap<String,ArrayList<Double>>();
    ArrayList<Double> queryVec=new ArrayList<Double>();
    ArrayList<Double> docVec=new ArrayList<Double>();

    for(String word:tfQuery.keySet()){
        queryVec.add(tfQuery.get(word));
        for(String type:tfs.keySet()){
            double val=0;
            if(tfs.get(type).containsKey(word)){
                val=tfs.get(type).get(word);
            }
            if (typeVec.containsKey(type)) {
                typeVec.get(type).add(val);
            } else {
                ArrayList list=new ArrayList<Double>();
                list.add(val);
                typeVec.put(type,list);
            }
        }
    }
    //update typeVec
    for (String type:typeVec.keySet()){
        if (type.equals("url")){
            for(int i=0;i<typeVec.get(type).size();i++){
                typeVec.get(type).set(i,
                        typeVec.get(type).get(i)*urlweight);
            }
        }else if (type.equals("title")){
            for(int i=0;i<typeVec.get(type).size();i++){
                typeVec.get(type).set(i,
                        typeVec.get(type).get(i)*titleweight);
            }

        }else if (type.equals("body")){
            for(int i=0;i<typeVec.get(type).size();i++){
                typeVec.get(type).set(i,
                        typeVec.get(type).get(i)*bodyweight);
            }

        }else if (type.equals("header")){
            for(int i=0;i<typeVec.get(type).size();i++){
                typeVec.get(type).set(i,
                        typeVec.get(type).get(i)*headerweight);
            }

        } else {
            //this is for anchor
            for(int i=0;i<typeVec.get(type).size();i++){
                typeVec.get(type).set(i,
                        typeVec.get(type).get(i)*anchorweight);
            }
        }
    }
    //interim doc vector
    for(String type:typeVec.keySet()){
        docVec=vectorOperation("add",docVec,typeVec.get(type));
    }

    for(double dbl: vectorOperation("multiply",docVec,queryVec))
        score+=dbl;
    return score;
  }
  
  /**
   * Normalize the term frequencies. 
   * @param tfs the term frequencies
   * @param d the Document
   * @param q the Query
   */
  public void normalizeTFs(Map<String,Map<String, Double>> tfs,Document d, Query q) {
    double length=smoothingBodyLength*(d.body_length+500); //body length normalization and smoothing

    for(String type:TFTYPES){
        for (String word:new HashSet<String>(q.queryWords)){
            if (tfs.containsKey(type)) {
                if (tfs.get(type).containsKey(word)) {
                    HashMap<String, Double> hmap = new HashMap<String, Double>();
                    hmap = (HashMap<String, Double>) tfs.get(type);
                    hmap.put(word, hmap.get(word) / length);
                }
            }
        }
    }
  }
  
  /**
   * Write the tuned parameters of cosineSimilarity to file.
   * Only used for grading purpose, you should NOT modify this method.
   * @param filePath the output file path.
   */
  private void writeParaValues(String filePath) {
    try {
      File file = new File(filePath);
      if (!file.exists()) {
        file.createNewFile();
      }
      FileWriter fw = new FileWriter(file.getAbsoluteFile());
      String[] names = {
        "urlweight", "titleweight", "bodyweight", "headerweight", 
        "anchorweight", "smoothingBodyLength"
      };
      double[] values = {
        this.urlweight, this.titleweight, this.bodyweight, 
    this.headerweight, this.anchorweight, this.smoothingBodyLength
      };
      BufferedWriter bw = new BufferedWriter(fw);
      for (int idx = 0; idx < names.length; ++ idx) {
        bw.write(names[idx] + " " + values[idx]);
        bw.newLine();
      }
      bw.close();
    } catch (IOException e) {
      e.printStackTrace();
    }
  }

  @Override
  /** Get the similarity score between a document and a query.
   * @param d the Document
   * @param q the Query
   * @return the similarity score.
   */
  public double getSimScore(Document d, Query q) {
    Map<String,Map<String, Double>> tfs = this.getDocTermFreqs(d,q);
    this.normalizeTFs(tfs, d, q);
    Map<String,Double> tfQuery = getQueryFreqs(q);

    // Write out tuned cosineSimilarity parameters
    // This is only used for grading purposes.
    // You should NOT modify the writeParaValues method.
    writeParaValues("cosinePara.txt");
    return getNetScore(tfs,q,tfQuery,d);
  }

  private ArrayList<Double> vectorOperation(String operation, ArrayList<Double> list1,ArrayList<Double> list2){
      ArrayList<Double> result=new ArrayList<Double>();
      if (operation.equals("add")){
          if (list1.size()==0)
              return list2;
          else {
              if (list1.size()!=list2.size())
                  return null;
              else {
                  for (int i = 0; i < list1.size(); i++) {
                      result.add(i, list1.get(i) + list2.get(i));
                  }
                  return result;
              }

          }
      } else {
          //this is for multiply
          if (list1.size()!=list2.size()){
              return null;
          } else {
              for (int i=0;i<list1.size();i++){
                  result.add(i,list1.get(i)*list2.get(i));
              }
              return result;
          }
      }
  }
}
