package cs276.pa4;

import java.io.*;
import java.util.*;

/**
 * This class is used to
 * 1) load training data from files
 * 2) build idf from data collections in PA1.
 */
public class LoadHandler {
  /** 
   * Loads the training data.
   * @param feature_file_name the name of the feature file.
   * @return the mapping of Query-url-Document
   */
  public static Map<Query,Map<String, Document>> loadTrainData(String feature_file_name) throws Exception {
    File feature_file = new File(feature_file_name);
    if (!feature_file.exists() ) {
      System.err.println("Invalid feature file name: " + feature_file_name);
      return null;
    }
    
    BufferedReader reader = new BufferedReader(new FileReader(feature_file));
    String line = null, url= null, anchor_text = null;
    Query query = null;
    
    /* Feature dictionary: Query -> (url -> Document)  */
    Map<Query,Map<String, Document>> queryDict =  new HashMap<Query,Map<String, Document>>();
    
    while ((line = reader.readLine()) != null) {
      String[] tokens = line.split(":", 2);
      String key = tokens[0].trim();
      String value = tokens[1].trim();

      if (key.equals("query")) {
        query = new Query(value);
        queryDict.put(query, new HashMap<String, Document>());
      } 
      else if (key.equals("url")) {
        url = value;
        queryDict.get(query).put(url, new Document(url));
      } 
      else if (key.equals("title")) {
        queryDict.get(query).get(url).title = new String(value.toLowerCase());
      }
      else if (key.equals("header")) {
        if (queryDict.get(query).get(url).headers == null)
          queryDict.get(query).get(url).headers =  new ArrayList<String>();
        queryDict.get(query).get(url).headers.add(value.toLowerCase());
      }
      else if (key.equals("body_hits")) {
        if (queryDict.get(query).get(url).body_hits == null)
          queryDict.get(query).get(url).body_hits = new HashMap<String, List<Integer>>();
        String[] temp = value.split(" ", 2);
        String term = temp[0].trim().toLowerCase();
        List<Integer> positions_int;
        
        if (!queryDict.get(query).get(url).body_hits.containsKey(term)) {
          positions_int = new ArrayList<Integer>();
          queryDict.get(query).get(url).body_hits.put(term, positions_int);
        } else
          positions_int = queryDict.get(query).get(url).body_hits.get(term);
        
        String[] positions = temp[1].trim().split(" ");
        for (String position : positions)
          positions_int.add(Integer.parseInt(position));
        
      } 
      else if (key.equals("body_length"))
        queryDict.get(query).get(url).body_length = Integer.parseInt(value);
      else if (key.equals("pagerank"))
        queryDict.get(query).get(url).page_rank = Integer.parseInt(value);
      else if (key.equals("anchor_text")) {
        anchor_text = value.toLowerCase();
        if (queryDict.get(query).get(url).anchors == null)
          queryDict.get(query).get(url).anchors = new HashMap<String, Integer>();
      }
      else if (key.equals("stanford_anchor_count"))
        queryDict.get(query).get(url).anchors.put(anchor_text, Integer.parseInt(value));      
    }

    reader.close();
    
    return queryDict;
  }
  
  /** 
   * Unserializes the term-doc counts from file.
   * @param idfFile the file containing the idfs.
   * @return the mapping of term-doc counts.
   */
  public static Map<String,Double> loadDFs(String idfFile) {
    Map<String,Double> termDocCount = null;
    try {
      FileInputStream fis = new FileInputStream(idfFile);
      ObjectInputStream ois = new ObjectInputStream(fis);
      termDocCount = (HashMap<String,Double>) ois.readObject();
      ois.close();
      fis.close();
    }
    catch(IOException | ClassNotFoundException ioe) {
      ioe.printStackTrace();
      return null;
    }
    return termDocCount;
  }
  
  /**
   * Builds document frequencies and then serializes to file.
   * @param dataDir the data directory
   * @param idfFile the file containing the idfs.
   * @return the term-doc counts
   */
  public static Map<String,Double> buildDFs(String dataDir, String idfFile) {
    // Get root directory
    String root = dataDir;
    File rootdir = new File(root);
    if (!rootdir.exists() || !rootdir.isDirectory()) {
      System.err.println("Invalid data directory: " + root);
      return null;
    }

    // Array of all the blocks (sub directories) in the PA1 corpus
    File[] dirlist = rootdir.listFiles();

    int totalDocCount = 0;

    // Count number of documents in which each term appears
    Map<String,Double> termDocCount = new HashMap<String, Double>();
    for(File d:dirlist){
      if (d.isDirectory()){
          File[] files=d.listFiles();
          for(File f:files){
              HashSet<String> words=new HashSet<>();
              try {
                  BufferedReader reader = new BufferedReader(new FileReader(f));
                  String line=null;
                  while((line=reader.readLine())!=null){
                      String[] tokens=line.trim().split("\\s+");
                      //convert to lower case
                      for(int i=0;i<tokens.length;i++){tokens[i]=tokens[i].toLowerCase();}
                      words.addAll(Arrays.asList(tokens));
                  }
                  reader.close();
              } catch (IOException e){e.printStackTrace();}
              for(String word:words){
                  if (termDocCount.containsKey(word))
                      termDocCount.put(word,termDocCount.get(word)+1);
                  else
                      termDocCount.put(word,1D);
              }
              totalDocCount++;
          }
      }
    }

    // Compute inverse document frequencies using document frequencies
    for (String term : termDocCount.keySet()) {
      //we are building the termDocCount from the corpus and the words are always going to be there
      termDocCount.put(term,Math.log(totalDocCount/termDocCount.get(term)));
    }
    
    // Save to file
    try {
      FileOutputStream fos = new FileOutputStream(idfFile);
      ObjectOutputStream oos = new ObjectOutputStream(fos);
      oos.writeObject(termDocCount);
      oos.close();
      fos.close();
    } catch(IOException ioe) {
      ioe.printStackTrace();
    }
    return termDocCount;
  }

}
