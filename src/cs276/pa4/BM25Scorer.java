package cs276.pa4;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.net.MalformedURLException;
import java.net.URL;
import java.util.*;

/**
 * Skeleton code for the implementation of a BM25 Scorer in Task 2.
 */
public class BM25Scorer extends AScorer {

  /*
   *  TODO: You will want to tune these values
   */
//  double urlweight = 0.1;
//  double titleweight  = 0.1;
//  double bodyweight = 0.1;
//  double headerweight = 0.1;
//  double anchorweight = 0.1;

  //take the cosine similarity scores

//  double urlweight = 0.9;
//  double titleweight  = 0.7;
//  double bodyweight = 0.3;
//  double headerweight = 0.6;
//  double anchorweight = 0.5;
//  double smoothingBodyLength = 1.0;

  // BM25-specific weights
//  double burl = 0.1;
//  double btitle = 0.1;
//  double bheader = 0.1;
//  double bbody = 0.1;
//  double banchor = 0.1;
//
//  double k1 = 0.1;
//  double pageRankLambda = 0.1;
//  double pageRankLambdaPrime = 0.1;


    double urlweight = 0.3;
    double titleweight  = 0.6;
    double bodyweight = 0.3;
    double headerweight = 0.8;
    double anchorweight = 0.5;
    double smoothingBodyLength = 1.0;

    /////// BM25 specific weights ///////////
    double burl=.75;
    double btitle=1.0;
    double bheader=.2;
    double bbody=.8;
    double banchor=0.8;


    double k1=1.0;
    double pageRankLambda=2.0;
    double pageRankLambdaPrime=-.5;
  
  // query -> url -> document
  Map<Query,Map<String, Document>> queryDict; 

  // BM25 data structures--feel free to modify these
  // Document -> field -> length
  Map<Document,Map<String,Double>> lengths;  

  // field name -> average length
  Map<String,Double> avgLengths;    

  // Document -> pagerank score
  Map<Document,Double> pagerankScores;

  // Term -> Score
  Map<String,Double> termScore = new HashMap<>();


    /**
     * Construct a BM25Scorer.
     * @param idfs the map of idf scores
     * @param queryDict a map of query to url to document
     */
    public BM25Scorer(Map<String,Double> idfs, Map<Query,Map<String, Document>> queryDict) {
      super(idfs);
      this.queryDict = queryDict;
      this.calcAverageLengths();
    }

    /**
     * Set up average lengths for BM25, also handling PageRank.
     */
  public void calcAverageLengths() {
    lengths = new HashMap<Document,Map<String,Double>>();
    avgLengths = new HashMap<String,Double>();
    //pagerankScores = new HashMap<Document,Double>();




    for (Map.Entry<Query,Map<String, Document>> entry: queryDict.entrySet()) {
        Map<String, Document> urlMap = entry.getValue();
        for (Map.Entry<String, Document> url: urlMap.entrySet()) {
            Document d = url.getValue();

            // for each document calculate the lengths for each attribute & put in the data structure.
            if (!lengths.containsKey(d)){
                // populate lengths d -> lengths
                calculateLengthForDocument(d);
                // populate pageranks
                //pagerankScores.put(d,
                //        Math.log(d.page_rank + pageRankLambdaPrime));
            }
        }
    }


    // calculate avg lengths...
    for (String tfType : this.TFTYPES) {
        double tScore = 0.0;
        //go over the all the documents and take the average of the scores..
        for(Map.Entry<Document,Map<String,Double>> entry: lengths.entrySet()){
            Map<String,Double> value = entry.getValue();
            tScore = tScore + value.get(tfType);
        }
        tScore = tScore/lengths.size();
        avgLengths.put(tfType, tScore);
    }

  }

  /// for each document and field type calculate the length
  private void calculateLengthForDocument(Document d){
      Map<String,Double> fields = new HashMap<String,Double>();
      for (String tfType : this.TFTYPES) {
          double size = getLength(d,tfType);
          fields.put(tfType,size);
      }
      // document -> field_name -> length map..
      //could even use pagerank map here too....
      lengths.put(d,fields);
  }

  /**
   * Get the net score. 
   * @param tfs the term frequencies
   * @param q the Query 
   * @param tfQuery
   * @param d the Document
   * @return the net score
   */
  public double getNetScore(Map<String,Map<String, Double>> tfs,
                            Query q, Map<String,Double> tfQuery,Document d) {

    double score = 0.0;
    //double pageRank = pagerankScores.get(d);
    termScore = getCombinedScore(tfs, q.queryWords);
    for (String term: q.queryWords) {
        // get idf value for the term...
        double idf = getIDFValue(term);
        double wdt = termScore.get(term);
        double numerator = wdt * idf;
        double denominator = wdt + k1;
        score+= numerator/denominator;

    }

    //score+=pageRankLambda*pageRank;
    return score;

  }

    private double getIDFValue(String term) {
        double idfScore = 0;
        if(!idfs.containsKey(term)){
            //ask...
            idfScore = 1;
//            return 1;
        }else{
            idfScore = idfs.get(term);
        }
        return idfScore;
    }

    private HashMap<String, Double> getCombinedScore(Map<String,Map<String, Double>> tfs, List<String> terms) {
        HashMap<String,Double> combinedVector = new HashMap<String,Double>();
        for(String word: terms){
            double score = 0;
            for(String type: TFTYPES){
                Map<String,Double> entry = tfs.get(type);
                if(entry.containsKey(word)){
                    score = score + entry.get(word);
                }
            }
            combinedVector.put(word, score);
        }
        return combinedVector;
    }

    /**
   * Do BM25 Normalization.
   * @param tfs the term frequencies
   * @param d the Document
   * @param q the Query
   */
  public void normalizeTFs(Map<String,Map<String, Double>> tfs,Document d, Query q) {

//      double length=smoothingBodyLength*(d.body_length+500); //body length normalization and smoothing


    for(Map.Entry<String,Map<String,Double>> entry : tfs.entrySet()){
      //Key
      String key = entry.getKey();
      //length of key in the document...
      double len = getLength(d,key);
      //avg length .. precomputed already...
      double avgLen = avgLengths.get(key);
      //QueryTerm -> Value...
      Map<String,Double> value = tfs.get(key);
      for(Map.Entry<String,Double> docEntry : value.entrySet()){
        // numerator of equation #3 - tf
        double numerator = docEntry.getValue();
        double denominator = 1 + getBWeight(key)*((len/avgLen) - 1);
        double weightedFT = getWeight(key)*(numerator/denominator);
        docEntry.setValue(weightedFT);
      }
    }
  }

  // BM25F weights...
  private double getBWeight(String key) {
      switch (key) {
          case "url":
              return burl;
          case "title":
              return btitle;
          case "body":
              return bbody;
          case "header":
              return bheader;
      }
    return 0;
  }


  // Normal Weights..
    private double getWeight(String key) {
        switch (key) {
            case "url":
                return urlweight;
            case "title":
                return titleweight;
            case "body":
                return bodyweight;
            case "header":
                return headerweight;
        }

        return 0;
    }

    // returns length of the document in the key (zone)
  private double getLength(Document d, String key) {

    double size = 0.0;
    if(key.equals("url")){
//      size = urlLength(d);
        URL url= null;
        try {
            url = new URL(d.url);
//            System.out.println(url);
            HashSet<String> hset=new HashSet<String>();
            hset.addAll(Arrays.asList(url.getHost().split("\\.")));
            hset.addAll(Arrays.asList(url.getPath().split("\\.")[0].split("/")));
            return hset.size();
        } catch (MalformedURLException e) {
            e.printStackTrace();
        }
      ;
    }
    if(key.equals("title")){
      String title = d.title;
      if(title != null){
        size = title.split("\\s+").length;
      }
    }
    if(key.equals("body")){
        // is this okay...
      size = d.body_length;
    }
    if(key.equals("header")){
      //ask should this be the number of items in header
        // or the number of items across all headers..
      List<String> headers = d.headers;
      if (headers!= null) {
          for (String header: headers) {
              size+= header.split("\\s+").length;
          }
      }

    }
    if(key.equals("anchor")){

        //same question..
        if (d.anchors!=null) {
            for (String text:d.anchors.keySet()){
                size+=text.split("\\s+").length;
            }
        }

    }
    return size;
  }

  /**
   * Write the tuned parameters of BM25 to file.
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
        "urlweight", "titleweight", "bodyweight", 
        "headerweight", "anchorweight", "burl", "btitle", 
        "bheader", "bbody", "banchor", "k1", "pageRankLambda", "pageRankLambdaPrime"
      };
      double[] values = {
        this.urlweight, this.titleweight, this.bodyweight, 
        this.headerweight, this.anchorweight, this.burl, this.btitle, 
        this.bheader, this.bbody, this.banchor, this.k1, this.pageRankLambda, 
        this.pageRankLambdaPrime
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
  /**
   * Get the similarity score.
   * @param d the Document
   * @param q the Query
   * @return the similarity score
   */
  public double getSimScore(Document d, Query q) {
    Map<String,Map<String, Double>> tfs = this.getDocTermFreqs(d,q);
    this.normalizeTFs(tfs, d, q);

    // ascorer -- done
    Map<String,Double> tfQuery = getQueryFreqs(q);

    // Write out the tuned BM25 parameters
    // This is only used for grading purposes.
    // You should NOT modify the writeParaValues method.
    //writeParaValues("bm25Para.txt");
    return getNetScore(tfs,q,tfQuery,d);
  }
  
}
