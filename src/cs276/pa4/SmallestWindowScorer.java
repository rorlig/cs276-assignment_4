package cs276.pa4;

import java.net.MalformedURLException;
import java.net.URL;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Map;

/**
 * A skeleton for implementing the Smallest Window scorer in Task 3.
 * Note: The class provided in the skeleton code extends BM25Scorer in Task 2. However, you don't necessarily
 * have to use Task 2. (You could also use Task 1, in which case, you'd probably like to extend CosineSimilarityScorer instead.)
 * Also, feel free to modify or add helpers inside this class.
 */
public class SmallestWindowScorer extends CosineSimilarityScorer {
  
  public SmallestWindowScorer(Map<String, Double> idfs) {
    super(idfs);
  }

  /**
   * get smallest window of one document and query pair.
   * @param d: document
   * @param q: query
   */  
  private int getWindow(Document d, Query q) {
      int smallestWindow=Integer.MAX_VALUE;
      HashSet<String> uniqTokens=new HashSet<String>();
      uniqTokens.addAll(q.queryWords);
      try {
          //URL parsing
          URL url=new URL(d.url);
          HashSet<String> hset=new HashSet<String>();
          hset.addAll(Arrays.asList(url.getHost().split(".")));
          for(String elem:url.getPath().split("/")){
              if (elem.indexOf('.')!=-1)
                  elem=elem.split("\\.")[0];
              hset.add(elem);
          }
          if(hset.containsAll(uniqTokens))
              smallestWindow=hset.size();
          hset.clear();
          //title
          if (d.title!=null) {
              hset.addAll(Arrays.asList(d.title.split(" ")));
              if (hset.containsAll(uniqTokens) && hset.size() < smallestWindow)
                  smallestWindow = hset.size();
          }
          hset.clear();
          /*
          //bodyhits (let's validate)
          if (d.body_hits!=null) {
              hset.addAll(d.body_hits.keySet());
              if (hset.containsAll(uniqTokens) && hset.size() < smallestWindow)
                  smallestWindow = hset.size();
          }
          hset.clear();
          */
          //anchor text
          if (d.anchors!=null) {
              for (String text : d.anchors.keySet()) {
                  hset.addAll(Arrays.asList(text.split(" ")));
                  if (hset.containsAll(uniqTokens) && hset.size() < smallestWindow)
                      smallestWindow = hset.size();
                  hset.clear();
              }
          }
          //headers
          if (d.headers!=null) {
              for (String header : d.headers) {
                  hset.addAll(Arrays.asList(header.split(" ")));
                  if (hset.containsAll(uniqTokens) && hset.size() < smallestWindow)
                      smallestWindow = hset.size();
                  hset.clear();
              }
          }
      } catch (MalformedURLException e) {
          e.printStackTrace();
      }

    return smallestWindow;
  }

  
  /**
   * get boost score of one document and query pair.
   * @param d: document
   * @param q: query
   */  
  private double getBoostScore (Document d, Query q) {
    int smallestWindow = getWindow(d, q);
    HashSet<String> uniqTokens=new HashSet<String>();
    uniqTokens.addAll(q.queryWords);
    double boostScore = 0;
    if (getWindow(d,q)==Integer.MAX_VALUE)
      boostScore=1;
    else
        boostScore=(smallestWindow+uniqTokens.size())/smallestWindow;
    return Math.pow(5,boostScore);
  }
  
  @Override
  public double getSimScore(Document d, Query q) {
    Map<String,Map<String, Double>> tfs = this.getDocTermFreqs(d,q);
    this.normalizeTFs(tfs, d, q);
    Map<String,Double> tfQuery = getQueryFreqs(q);
    double boost = getBoostScore(d, q);
    double rawScore = this.getNetScore(tfs, q, tfQuery, d);
    return boost * rawScore;
  }

}
