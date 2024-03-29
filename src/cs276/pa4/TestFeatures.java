package cs276.pa4;

import weka.core.Instances;

import java.util.Map;

/**
 * A sample class to store the result
 */
public class TestFeatures {
    /* Test features */
    Instances features;

    /* Associate query-doc pair to its index within FEATURES instances
     * {query -> {doc -> index}}
     *
     * For example, you can get the feature for a pair of (query, doc) using:
     *   features.get(index_map.get(query).get(doc));
     * */
    Map<Query, Map<Document, Integer>> index_map;
    Map<Query, Map<Pair<Document, Document>, Integer>> pairwise_index_map;

    /*
    {q1:{<d1,d2>:1, <d2,d3>:2, <d3,d4>:3}, q1 -> d1,d2,d3

        q2:{<d5,d6>:4, <d7:d8>:5}}
        */
}
