package cs276.pa4;


import java.net.MalformedURLException;
import java.net.URL;
import java.util.*;

public class Feature {

    public static boolean isSublinearScaling = true;
    private Parser parser = new Parser();
    double smoothingBodyLength = 800;

    Map<String, Double> idfs;

    // If you would like to use additional features, you can declare necessary variables here
  /*
   * @TODO: Your code here
   */

    public Feature(Map<String, Double> idfs) {
        this.idfs = idfs;
    }


    public double[] extractFeatureVector(Document d, Query q) {
    
    /* Compute doc_vec and query_vec */
        Map<String, Map<String, Double>> tfs = Util.getDocTermFreqs(d, q);
        Map<String, Double> queryVector = getQueryVec(q);

        // normalize term-frequency
        this.normalizeTFs(tfs, d, q);
    
    /* [url, title, body, header, anchor] */
        double[] result = new double[5];
        for (int i = 0; i < result.length; i++) {
            result[i] = 0.0;
        }
        for (String queryWord : q.queryWords) {
            double queryScore = queryVector.get(queryWord);
            result[0] += tfs.get("url").get(queryWord) * queryScore;
            result[1] += tfs.get("title").get(queryWord) * queryScore;
            result[2] += tfs.get("body").get(queryWord) * queryScore;
            result[3] += tfs.get("header").get(queryWord) * queryScore;
            result[4] += tfs.get("anchor").get(queryWord) * queryScore;
        }

        return result;
    }

    public double[] addPreictedVarToFeatureVec(double[] covs, double val) {
        double[] vector = new double[covs.length + 1];
        for (int i = 0; i < covs.length; i++) {
            vector[i] = covs[i];
        }
        vector[covs.length] = val;
        return vector;
    }

    /* Generate query vector */
    public Map<String, Double> getQueryVec(Query q) {
    /* Count word frequency within the query, in most cases should be 1 */

        Map<String, Double> tfVector = new HashMap<String, Double>();
        String[] wordInQuery = q.query.toLowerCase().split(" ");
        for (String word : wordInQuery) {
            if (tfVector.containsKey(word))
                tfVector.put(word, tfVector.get(word) + 1);
            else
                tfVector.put(word, 1.0);
        }
    
    /* Sublinear Scaling */
        if (isSublinearScaling) {
            for (String word : tfVector.keySet()) {
                tfVector.put(word, 1 + Math.log(tfVector.get(word)));
            }
        }
    
    /* Compute idf vector */
        Map<String, Double> idfVector = new HashMap<String, Double>();

        for (String queryWord : q.queryWords) {
            if (this.idfs.containsKey(queryWord))
                idfVector.put(queryWord, this.idfs.get(queryWord));
            else {
                idfVector.put(queryWord, Math.log(98998.0)); /* Laplace smoothing */
            }

        }
    
    /* Do dot-product */
        Map<String, Double> queryVector = new HashMap<String, Double>();
        for (String word : q.queryWords) {
            queryVector.put(word, tfVector.get(word) * idfVector.get(word));
        }

        return queryVector;
    }

    public void normalizeTFs(Map<String, Map<String, Double>> tfs, Document d, Query q) {
        double normalizationFactor = (double) (d.body_length) + (double) (smoothingBodyLength);

        for (String queryWord : q.queryWords)
            for (String tfType : tfs.keySet())
                tfs.get(tfType).put(queryWord, tfs.get(tfType).get(queryWord) / normalizationFactor);
    }

    public double[] extractMoreFeatures(Document d, Query q, Map<Query, Map<String, Document>> dataMap) {

        double[] basic = extractFeatureVector(d, q);
        //double[] bm25=calculateBM25Weights(d,q);
        double[] others={d.page_rank,new SmallestWindowScorer(idfs).getSimScore(d,q),
                new BM25Scorer(idfs,dataMap).getSimScore(d,q)};
        double[] result=new double[basic.length+others.length];
        for(int i=0;i<basic.length;i++)
            result[i]=basic[i];
        for(int j=basic.length;j<result.length;j++)
            result[j]=others[j-basic.length];
        return result;
    }
    private double[] calculateBM25Weights(Document d, Query q){
        Map<String, Map<String, Double>> tfs = Util.getDocTermFreqs(d, q);
        Map<String, Double> queryVector = getQueryVec(q);
        this.normalizeTFs(tfs,d,q);
        int k=1;
        double[] scores=new double[5];
        double[] bm25Weights={0.75,1.0,0.2,0.8,0.8}; //url, title, body. header, anchor
        for (int i=0;i<scores.length;i++)
            scores[i]=0.0;
        for (String word:q.queryWords) {
            double queryScore = queryVector.get(word);
            scores[0] += queryScore*tfs.get("url").get(word)*(k+1)/(k+tfs.get("url").get(word)*queryScore);
            scores[1] += queryScore*tfs.get("title").get(word)*(k+1)/(k+tfs.get("title").get(word)*queryScore);
            scores[2] += queryScore*tfs.get("body").get(word)*(k+1)/(k+tfs.get("body").get(word)*queryScore);
            scores[3] += queryScore*tfs.get("header").get(word)*(k+1)/(k+tfs.get("header").get(word)*queryScore);
            scores[4] += queryScore*tfs.get("anchor").get(word)*(k+1)/(k+tfs.get("anchor").get(word)*queryScore);
        }
        for(int i=0;i<scores.length;i++)
            scores[i]*=bm25Weights[i];
        return scores;
    }
}
