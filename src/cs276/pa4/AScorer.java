package cs276.pa4;

import java.net.MalformedURLException;
import java.net.URL;
import java.util.*;

/**
 * An abstract class for a scorer. 
 * Needs to be extended by each specific implementation of scorers.
 */
public abstract class AScorer {

    // Map: term -> idf
    Map<String,Double> idfs;

    // Various types of term frequencies that you will need
    String[] TFTYPES = {"url","title","body","header","anchor"};

    /**
     * Construct an abstract scorer with a map of idfs.
     * @param idfs the map of idf scores
     */
    public AScorer(Map<String,Double> idfs) {
        this.idfs = idfs;
    }

    /**
     * You can implement your own function to whatever you want for debug string
     * The following is just an example to include page information in the debug string
     * The string will be forced to be 1-line and truncated to only include the first 200 characters
     */
    public String getDebugStr(Document d, Query q)
    {
        return "Rating: " + this.getSimScore(d,q);
    }

    /**
     * Score each document for each query.
     * @param d the Document
     * @param q the Query
     */
    public abstract double getSimScore(Document d, Query q);

    /**
     * Get frequencies for a query.
     * @param q the query to compute frequencies for
     */
    public Map<String,Double> getQueryFreqs(Query q) {

        // queryWord -> term frequency
        Map<String,Double> tfQuery = new HashMap<String, Double>();

        //weight function - wf = 1+log(tf) if tf>0 else 0; wf'=wf*idf;
        List<String> tokens=q.queryWords;
        HashSet<String> uniqTokens=new HashSet<>();
        uniqTokens.addAll(tokens);
        for(String s:uniqTokens){
            if (this.idfs.keySet().contains(s))
                tfQuery.put(s,this.idfs.get(s)*
                    (1+Math.log(getCount(s,tokens))));
            else
                tfQuery.put(s,
                        (1+Math.log(getCount(s,tokens))));
        }
        return tfQuery;
    }


  /*
   * TODO : Your code here
   * Include any initialization and/or parsing methods
   * that you may want to perform on the Document fields
   * prior to accumulating counts.
   * See the Document class in Document.java to see how
   * the various fields are represented.
   */


    /**
     * Accumulate the various kinds of term frequencies
     * for the fields (url, title, body, header, and anchor).
     * You can override this if you'd like, but it's likely
     * that your concrete classes will share this implementation.
     * @param d the Document
     * @param q the Query
     */
    public Map<String,Map<String, Double>> getDocTermFreqs(Document d, Query q) {

        // Map from tf type -> queryWord -> score
        Map<String,Map<String, Double>> tfs = new HashMap<String,Map<String, Double>>();
        for(String str:TFTYPES)
            tfs.put(str,new HashMap<String,Double>());

        //query words are wrapped around a set for uniqueness
        for (String queryWord : new HashSet<String>(q.queryWords)) {

            try {
                //url
                if (d.url!=null) {
                    URL u = new URL(d.url);
                    HashSet<String> hset = new HashSet<>();
                    hset.addAll(Arrays.asList(u.getHost().split(".")));
                    hset.addAll(Arrays.asList(u.getPath().split("/")));
                    for (String s : hset) {
                        if (s.indexOf('.')!=-1){
                            s=s.split("\\.")[0];

                        }
                        if (s.equals(queryWord))
                            if (tfs.get("url").containsKey(queryWord))
                                tfs.get("url").put(queryWord, tfs.get("url").get(queryWord) + 1);
                            else
                                tfs.get("url").put(queryWord, 1D);
                    }
                }
                //title
                if (d.title!=null) {
                    for (String s : d.title.split(" ")) {
                        if (s.equals(queryWord))
                            if (tfs.get("title").containsKey(queryWord))
                                tfs.get("title").put(queryWord, tfs.get("title").get(queryWord) + 1);
                            else
                                tfs.get("title").put(queryWord, 1D);
                    }
                }
                //headers
                if (d.headers!=null) {
                    HashSet<String> headerWords = new HashSet<String>();
                    for (String header : d.headers)
                        headerWords.addAll(Arrays.asList(header.split(" ")));
                    for (String s : headerWords) {
                        if (s.equals(queryWord))
                            if (tfs.get("header").containsKey(queryWord))
                                tfs.get("header").put(queryWord, tfs.get("header").get(queryWord) + 1);
                            else
                                tfs.get("header").put(queryWord, 1D);
                    }
                }
                //bodyhits
                if (d.body_hits!=null) {
                    for (String key : d.body_hits.keySet()) {
                        if (key.equals(queryWord))
                            if (tfs.get("body").containsKey(queryWord))
                                tfs.get("body").put(queryWord, tfs.get("body").get(queryWord) + d.body_hits.get(key).size());
                            else
                                tfs.get("body").put(queryWord, (double) d.body_hits.get(key).size());
                    }
                }
                //anchors
                if (d.anchors!=null) {
                    for (String atext : d.anchors.keySet()) {
                        for (String s : atext.split(" ")) {
                            if (s.equals(queryWord))
                                if (tfs.get("anchor").containsKey(queryWord))
                                    tfs.get("anchor").put(queryWord, tfs.get("anchor").get(queryWord) + d.anchors.get(atext));
                                else
                                    tfs.get("anchor").put(queryWord, (double) d.anchors.get(atext));
                        }
                    }
                }

            } catch (MalformedURLException e) {
                e.printStackTrace();
            }

        }
        return tfs;
    }

    public int getCount(String str, List<String> list){
        int count=0;
        if (!list.contains(str))
            return count;
        for(String s:list)
            if (s.equals(str))
                count++;
        return count;
    }

}