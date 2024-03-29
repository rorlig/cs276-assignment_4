package cs276.pa4;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import weka.classifiers.Classifier;
import weka.core.Instances;

/**
 * Main entry-point of PA4
 * Version 2.0: includes idfs_file as a command line argument
 */
public class Learning2Rank {

    /**
     * Returns a trained model
     *
     * @param train_signal_file
     * @param train_rel_file
     * @param task              1: Linear Regression
     *                          2: SVM
     *                          3: More features
     *                          4: Extra credit
     * @param idfs
     * @return
     */
    public static Classifier train(String train_signal_file, String train_rel_file, int task, Map<String, Double> idfs, double c,
                                   double gamma) {
        System.err.println("## Training with feature_file =" + train_signal_file + ", rel_file = " + train_rel_file + " ... \n");
        Classifier model = null;
        Learner learner = null;

        if (task == 1) {
            learner = new PointwiseLearner();
        } else if (task == 2) {
            boolean isLinearKernel = false;
            learner = new PairwiseLearner(c, gamma, isLinearKernel);
        } else if (task == 3) {
            boolean isLinearKernel = false;
            learner = new AdvancedLearner(c, gamma, isLinearKernel);
      /* 
       * @TODO: Your code here, add more features 
       * */
            System.err.println("Task 3");

        } else if (task == 4) {
      
      /* 
       * @TODO: Your code here, extra credit 
       * */
            System.err.println("Extra credit");

        }
    
    /* Step (1): construct your feature matrix here */
        Instances data = learner.extractTrainFeatures(train_signal_file, train_rel_file, idfs);
    
    /* Step (2): implement your learning algorithm here */
        model = learner.training(data);

        return model;
    }

    /**
     * Test model using the test signal file
     *
     * @param test_signal_file
     * @param model
     * @param task
     * @param idfs
     * @return
     */
    public static Map<Query, List<Document>> test(String test_signal_file, Classifier model, int task, Map<String, Double> idfs,
                                                  double c, double gamma) {
        System.err.println("## Testing with feature_file=" + test_signal_file + " ... \n");
        Map<Query, List<Document>> ranked_queries = new HashMap<Query, List<Document>>();
        Learner learner = null;
        if (task == 1) {
            learner = new PointwiseLearner();
        } else if (task == 2) {
            boolean isLinearKernel = false;
            learner = new PairwiseLearner(c, gamma, isLinearKernel);
        } else if (task == 3) {
            boolean isLinearKernel = false;
            learner = new AdvancedLearner(c, gamma, isLinearKernel);
      /* 
       * @TODO: Your code here, add more features 
       * */
            System.err.println("Task 3");

        } else if (task == 4) {
       
      /* 
       * @TODO: Your code here, extra credit 
       * */
            System.err.println("Extra credit");

        }
    /* Step (1): construct your test feature matrix here */
        TestFeatures tf = learner.extractTestFeatures(test_signal_file, idfs);
    
    /* Step (2): implement your prediction and ranking code here */
        ranked_queries = learner.testing(tf, model);

        return ranked_queries;
    }


    /**
     * Output the ranking results in expected format
     *
     *
     * @param ps
     */
    public static void writeRankedResultsToFile(Map<Query, List<Document>> queryRankings, PrintStream ps) {
        for (Query query : queryRankings.keySet()) {
            StringBuilder queryBuilder = new StringBuilder();
            for (String s : query.queryWords) {
                queryBuilder.append(s);
                queryBuilder.append(" ");
            }

            String queryStr = "query: " + queryBuilder.toString() + "\n";
            ps.print(queryStr);

            for (Document res : queryRankings.get(query)) {
                String urlString =
                        "  url: " + res.url + "\n" +
                                "    title: " + res.title + "\n" +
                                "    debug: " + res.debugStr + "\n";
                ps.print(urlString);
            }
        }
    }

    public static void main(String[] args) throws IOException {
        if (args.length != 5 && args.length != 6) {
            System.err.println("Input arguments: " + Arrays.toString(args));
            System.err.println("Usage: <train_signal_file> <train_rel_file> <test_signal_file> <idfs_file> <task> [ranked_out_file]");
            System.err.println("  ranked_out_file (optional): output results are written into the specified file. If not, output to stdout.");
            return;
        }
        String train_signal_file = args[0];
        String train_rel_file = args[1];
        String test_signal_file = args[2];
        String dfFile = args[3];
        int task = Integer.parseInt(args[4]);
        String ranked_out_file = "";
        if (args.length == 6) {
            ranked_out_file = args[5];
        }
    
    /* Populate idfs */
        Map<String, Double> idfs = Util.loadDFs(dfFile);
        double best_c  = 1.0, best_gamma = 0.25; //best c and gamma
        if (task == 1 || task == 3) {
            train_and_test(train_signal_file, train_rel_file, test_signal_file, 1, idfs, ranked_out_file, best_c, best_gamma);
        } else {
            for (double c= 8.0 ; c >= 0.125; c /=2) {
                for (double gamma = 0.5;  gamma >= 0.0078125; gamma/=2) {
                    train_and_test(train_signal_file, train_rel_file, test_signal_file, task, idfs, ranked_out_file, c, gamma);
                }
            }
        }

    }

    private static void train_and_test(String train_signal_file, String train_rel_file, String test_signal_file, int task, Map<String, Double> idfs, String ranked_out_file,
                                       double c, double gamma) throws IOException {
         /* Train & test */
        System.err.println("### Running task" + task + "..." + " gamma " + gamma + " c " + c);
        Classifier model = train(train_signal_file, train_rel_file, task, idfs, c, gamma);
    /* performance on the training data */
        Map<Query, List<Document>> trained_ranked_queries = test(train_signal_file, model, task, idfs, c, gamma);
        String trainOutFile = "tmp.train.ranked1";
        writeRankedResultsToFile(trained_ranked_queries, new PrintStream(new FileOutputStream(trainOutFile)));
        NdcgMain ndcg = new NdcgMain(train_rel_file);
        System.err.println("# Trained NDCG=" + ndcg.score(trainOutFile));
        //(new File(trainOutFile)).delete();

        Map<Query, List<Document>> ranked_queries = test(test_signal_file, model, task, idfs, c, gamma);

    /* Output results */
        if (ranked_out_file == null || ranked_out_file.isEmpty()) { /* output to stdout */
            writeRankedResultsToFile(ranked_queries, System.out);
        } else {
      /* output to file */
            try {
                writeRankedResultsToFile(ranked_queries, new PrintStream(new FileOutputStream(ranked_out_file)));
                ndcg = new NdcgMain("pa4-data/pa3.rel.dev");
                System.err.println("# Tested NDCG=" + ndcg.score(ranked_out_file));
            } catch (FileNotFoundException e) {
                e.printStackTrace();
            }
        }
    }
}

