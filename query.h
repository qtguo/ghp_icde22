#ifndef FHP_QUERY_H
#define FHP_QUERY_H

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cstdlib>

#include "algo.h"
#include "graph.h"
#include "heap.h"
#include "config.h"
#include "build.h"
#include "SparseMatrix.h"
#include <time.h>
#include <iomanip>
#include <unordered_set>
#include "group.h"

//#define CHECK_PPR_VALUES 1
// #define CHECK_TOP_K_PPR 1
#define PRINT_PRECISION_FOR_DIF_K 1
// std::mutex mtx;

double avg_num_larger_upper_bound = 0;
double avg_num_larger_lower_bound = 0;
double avg_num_iteraion = 0;
double avg_time = 0;
double backpush_time =0;
double second_phase_time = 0;
double first_phase_time =0;
vector<double> avg_prec_vec;
vector<double> avg_ndcg_vec;
vector<double> avg_num_larger_upper_bound_each_iter;
vector<double> avg_num_larger_lower_bound_each_iter;
vector<int> num_query_each_iter;
//vector<unordered_map<int, double>> residual_maps;//for each candidate, a residual map
vector<unordered_map<int, double>> visiting_prob_maps;//for each candidate, a visiting prob map
unordered_map<int, int> node_to_order;
unordered_map<int, double> upper_bounds_fhp;
unordered_map<int, double> lower_bounds_fhp;
vector<double> ppr_t_t_vec;

vector<vector<int>> groups_list;
vector<vector<int>> groups_inverted_list;
unordered_map<int, vector<int>> candidate_groups_inverted_map;

void create_dir(string& dir)
{
    struct stat statbuf;
    if (stat(dir.c_str(), &statbuf) != 0 || !S_ISDIR(statbuf.st_mode))
    {
        cout << "dir " << dir << " not exist ,creat it" <<endl;
        string cmd = "mkdir -p " + dir;
        std::system(cmd.c_str());
    }

}

string build_topk_result_file_path(int k, int source)
{
    string algo = config.algo;
    if(!config.sqrt_walk){
        algo = algo + "-" + "rw";
    }

    string dir = "estimated_fhp/" + config.graph_alias + "/topk/" + algo +"/" + to_string(k) + "/";
    create_dir(dir);
    return dir + to_str(source) + "_" + to_str(config.epsilon) +"_g" + to_str(config.gsize) + ".txt";
}

string build_topk_statistics_file_path()
{
    string algo = config.algo;
    if(!config.sqrt_walk){
        algo = algo + "-" + "rw";
    }
    string dir = "estimated_fhp/" + config.graph_alias + "/topk/" + algo +"/" ;
    create_dir(dir);
    return dir + to_str(config.epsilon) + "_gsize_" + to_str(config.gsize) + "_results.txt";
}


double mc_pairwise_fhp(int source, int target, const Graph& graph){
    //for pairwise query
    //rw_counter.reset_zero_values();
    //rw stops as long as hitting target
   /* stringstream ss;
    ss << "estimated_fhp/" << config.graph_alias << "/pairwise/mc/" << source << ".txt";
    string outfile = ss.str();
    cout<<outfile<<endl;

    ofstream est_ppr_file(outfile);
*/
    //clock_t start = clock();
    //for(int i=0; i<target_set.size(); i++){
    //    int target = target_set[i];
    //    cout<<i+1 <<"-th node: " << target<<endl;
        double fhp=0.0;
        clock_t start = clock();

        if(source == target){
            fhp = 1.0;
        }else {
            unsigned long rw_counter = 0;
            unsigned long num_rw = 3 * graph.n * log(2 * graph.n) / config.epsilon / config.epsilon;
            //cout << "# walks: " << num_rw << endl;

            for (unsigned long j = 0; j < num_rw; j++) {
                int temp_node = source;
                while (drand() > config.alpha) {
                    if (graph.g[temp_node].size()) {
                        int next = lrand() % graph.g[temp_node].size();
                        temp_node = graph.g[temp_node][next];

                        if (temp_node == target) {
                            rw_counter += 1;
                            break;
                        }

                    } else {
                        break;
                    }
                }
            }

            fhp = rw_counter / (double) num_rw;
        }
        //cout<<"estimated fhp by mc: " << fhp <<endl;
        //est_ppr_file<<target <<" " <<fhp<<endl;
    //}

    clock_t end = clock();

    avg_time += (end - start)/(double) CLOCKS_PER_SEC;

    return fhp;
}

void mc_topk_fhp(int source, const Graph& graph) {
    cout<<"here"<<endl;
    rw_counter.reset_zero_values();

    vector<double> fhps;
    vector<pair<int, double>> ordered_fhps;
    vector<bool> visited;
    for (int i = 0; i < graph.n; i++) {
        fhps.push_back(0.0);
        ordered_fhps.push_back(make_pair(i, 0.0));
        visited.push_back(false);
    }

    unsigned long long num_rw = 3 * graph.n * log(2 * graph.n) / config.epsilon /
                           config.epsilon; //for ground truth, epsilon should be 0.1 or 0.2 (smaller than 0.5)
    cout << "# walks: " << num_rw << endl;

    clock_t start = clock();
    for (unsigned long long j = 0; j < num_rw; j++) {
        //a random walk from source
        int temp_node = source;
        //int step = 0;
        vector<int> single_walk;
        single_walk.push_back(source);
        visited[source] = true;
        while (drand() > config.alpha) {
            // step += 1;
            // double incre = pow(sqrt(1 - config.alpha), step);
            if (graph.g[temp_node].size()) {
                int next = lrand() % graph.g[temp_node].size();
                temp_node = graph.g[temp_node][next];
                single_walk.push_back(temp_node);
                if (visited[temp_node] == false) {
                    if (!rw_counter.exist(temp_node))
                        rw_counter.insert(temp_node, 1);
                    else
                        rw_counter[temp_node] += 1;

                    visited[temp_node] = true;
                }
                /* if(temp_node == target){
                     break;

                 }
                 */

            } else {
                break;
            }

        }

        for (int l = 0; l < single_walk.size(); l++) {
            int visited_node = single_walk[l];
            visited[visited_node] = false;
        }

    }

    //estimate the fhp for all nodes
    cout << "estimating fhps " << endl;
    for (int i = 0; i < graph.n; i++) {
        if (rw_counter.exist(i) && i != source) {
            //cout<<"current i: " <<i<<endl;
            fhps[i] = rw_counter[i] / (double) num_rw;
            ordered_fhps[i] = make_pair(i, fhps[i]);
        }
        else if(i == source){
            ordered_fhps[i] = make_pair(i, 1.0);
        }
    }

    clock_t end = clock();

    avg_time += (end - start)/(double) CLOCKS_PER_SEC;
    //sorting the nodes in decreasing of fhps
    cout << "sorting fhps" << endl;
    sort(ordered_fhps.begin(), ordered_fhps.end(),
         [](pair<int, double> const &l, pair<int, double> const &r) { return l.second > r.second; });


    stringstream ss;
    ss << "estimated_fhp/" << config.graph_alias << "/topk/mc/"<<to_str(source)<< "_"<<to_str(config.epsilon) <<".txt";
    string outfile = ss.str();
    ofstream topk_file(outfile);

    for(int i=0; i<graph.n; i++){
        pair<int, double> ordered_fhp = ordered_fhps[i];
        topk_file<<ordered_fhp.first << " "<< ordered_fhp.second<<endl;
    }

    topk_file.close();


}

void mc_topk_group(int source, const Graph& graph) {
    rw_counter.reset_zero_values();

    vector<double> fhps(groups_list.size());
    vector<pair<int, double>> ordered_fhps;

    for (int i = 0; i < groups_list.size(); i++) {
        ordered_fhps.push_back(make_pair(i, 0.0));
    }

    unsigned long long num_rw = 3 * graph.n * log(2 * graph.n) / config.epsilon /
                           config.epsilon; //for ground truth, epsilon should be 0.1 or 0.2 (smaller than 0.5)
    cout << "# walks: " << num_rw << endl;

    clock_t start = clock();
    for (unsigned long long j = 0; j < num_rw; j++) {
        //a random walk from source
        int temp_node = source;
        //int step = 0;
        vector<int> single_walk;
        single_walk.push_back(source);
        while (drand() > config.alpha) {
            if (graph.g[temp_node].size()) {
                int next = lrand() % graph.g[temp_node].size();
                temp_node = graph.g[temp_node][next];
                single_walk.push_back(temp_node);

                /* if(temp_node == target){
                     break;

                 }
                 */

            } else {
                break;
            }

        }

        unordered_set<int> group_hitted_set;
        for (int l = 0; l < single_walk.size(); l++)
        {
            int visited_node = single_walk[l];
            for (auto group_index: groups_inverted_list[visited_node])
            {
                if (group_hitted_set.find(group_index) == group_hitted_set.end())
                {
                    if (!rw_counter.exist(group_index))
                    {
                        rw_counter.insert(group_index, 1);
                    }
                    else
                    {
                        rw_counter[group_index] += 1;
                    }
                    group_hitted_set.insert(group_index);
                }
            }
        }

    }

    //estimate the fhp for all nodes
    cout << "estimating fhps " << endl;
    for (int i = 0; i < groups_list.size(); i++) {
        if (rw_counter.exist(i)) {
            //cout<<"current i: " <<i<<endl;
            fhps[i] = rw_counter[i] / (double) num_rw;
            ordered_fhps[i] = make_pair(i, fhps[i]);
        }
    }

    clock_t end = clock();

    avg_time += (end - start)/(double) CLOCKS_PER_SEC;
    //sorting the nodes in decreasing of fhps
    cout << "sorting fhps" << endl;
    sort(ordered_fhps.begin(), ordered_fhps.end(),
         [](pair<int, double> const &l, pair<int, double> const &r) { return l.second > r.second; });

    string outfile = build_topk_result_file_path(config.k, source);;

    ofstream topk_file(outfile);

    for(int i=0; i<groups_list.size(); i++){
        pair<int, double> ordered_fhp = ordered_fhps[i];
        topk_file<<ordered_fhp.first << " "<< ordered_fhp.second<<endl;
    }

    topk_file.close();
}

void visiting_prob(int source, int target, const Graph& graph){
    //a walk stops, when it hit t

        double* vp=new double[graph.n];
        clock_t start = clock();

            unsigned long rw_counter = 0;
            unsigned long num_rw = 3 * graph.n * log(2 * graph.n) / config.epsilon / config.epsilon;
            cout << "# walks: " << num_rw << endl;

            for (unsigned long j = 0; j < num_rw; j++) {
                int temp_node = source;
                while (drand() > config.alpha) {
                    //double incre = pow(sqrt(1 - config.alpha), step);
                    //incre *= sqrt_value;

                    if (graph.g[temp_node].size()) {
                        int next = lrand() % graph.g[temp_node].size();
                        temp_node = graph.g[temp_node][next];

                        if(temp_node == target){
                            break;
                        }
                        else{
                            if(!sqrt_rw_counter.exist(temp_node))
                                sqrt_rw_counter.insert(temp_node, 1);
                            else
                                sqrt_rw_counter[temp_node] += 1;
                        }

                    }
                    else {
                        break;
                    }

                }
            }

            for(long j=0; j<graph.n; j++){
                // ppr[i]+=counts[residue.first]*1.0/config.omega*residue.second;
                int nodeid = j;

                if(nodeid != target){
                    int occur;
                    if(!sqrt_rw_counter.exist(nodeid))
                        occur = 0.0;
                    else
                        occur = sqrt_rw_counter[nodeid];

                    vp[nodeid] = occur*1.0/(double) num_rw;
                    cout<<"node id: " << nodeid << " " << vp[nodeid] <<endl;
                }
            }


}

void samba_query_pairwise_sqrt_walk(int source, vector<int> target_set, const Graph& graph) {
    //Timer timer(BIPPR_QUERY);
    stringstream ss;
    ss << "estimated_fhp/" << config.graph_alias << "/pairwise/samba/" << source << ".txt";
    string outfile = ss.str();
    cout<<outfile<<endl;
    ofstream est_ppr_file(outfile);

    clock_t start = clock();
    for(int i=0; i<target_set.size(); i++){
        int target = target_set[i];
        //sqrt_rw_counter.clean();
        sqrt_rw_counter.reset_zero_values();
        cout<<i+1 <<"-th node: " << target<<endl;
        double fhp = 0.0;
        //double stop_prob = 1 - sqrt(1-config.alpha);
        double stop_prob = config.alpha;
        cout<<"stop probability: " << stop_prob <<endl;

        unordered_set<int> set_out_neighbour_s;
        for(int i=0; i< graph.g[source].size(); i++){
            set_out_neighbour_s.insert(graph.g[source][i]);
        }

        sqrt_rw_counter.clean();
        if (source == target) {
            fhp = 1.0;
        }
        else if(set_out_neighbour_s.count(target)>0){
            cout<<"t is an out-neighbour of s" <<endl;
            int outdeg_source = graph.g[source].size();
            unsigned long num_rw = 3 * outdeg_source * log(2.0/config.pfail) / config.epsilon / config.epsilon / stop_prob /(1-config.alpha);
            cout<< "num rw: " << num_rw <<endl;
            double counter = 0.0;
            for (unsigned long j = 0; j < num_rw; j++) {
                int temp_node = source;
                int step = 0;
                int incre = 1;
                //double sqrt_value = sqrt(1-config.alpha);
                //double sqrt_value = 1.0;
                while (drand() > stop_prob) {
                    step += 1;
                    //double incre = pow(sqrt(1 - config.alpha), step);
                    //incre *= sqrt_value;
                    if (graph.g[temp_node].size()) {
                        int next = lrand() % graph.g[temp_node].size();
                        temp_node = graph.g[temp_node][next];

                        if(temp_node == target){
                            counter += incre;
                            break;
                        }
                    }
                    else {
                        break;
                    }

                }
            }
            fhp =  counter /(double) num_rw;
        }
        else{
            cout<<"t is not an out-neighbour of s" <<endl;
            //double new_rmax = 0.1*config.rmax; //10^{-6}
            INFO(config.rmax);
            //cout<< "rmax used: " << new_rmax <<endl;
            //first run the backward propagation from the target
            reverse_local_update_linear_visiting_prob(source, target, set_out_neighbour_s, graph, config.rmax);

            INFO(config.omega);
            //double total_X = 0.0;
            unsigned long total_visits = 0;
            //rw_counter.clean();
            for (unsigned long j = 0; j < config.omega; j++) {
                //a random walk from i
                //terminates when it hits t
                int temp_node = source;
                int step = 0;
                //double incre = 1.0;
                //double sqrt_value = sqrt(1-config.alpha);
                int incre = 1;
                //sqrt_rw_counter[source] += 1.0;
                while (drand() > stop_prob) {
                    step += 1;
                    //double incre = pow(sqrt(1 - config.alpha), step);
                    //incre *= sqrt_value;
                    if (graph.g[temp_node].size()) {
                        int next = lrand() % graph.g[temp_node].size();
                        temp_node = graph.g[temp_node][next];

                        if(temp_node == target){
                            break;
                        }

                        else{
                            if(!sqrt_rw_counter.exist(temp_node))
                                sqrt_rw_counter.insert(temp_node, incre);
                            else
                                sqrt_rw_counter[temp_node] += incre;
                        }

                    }
                    else{
                        break;
                    }

                }

               /* if(temp_node != target){
                    if(!sqrt_rw_counter.exist(temp_node))
                        sqrt_rw_counter.insert(temp_node, incre/(double) config.alpha);
                    else
                        sqrt_rw_counter[temp_node] += incre/(double) config.alpha;
                }
*/

            }
            for(long j=0; j<bwd_idx.second.occur.m_num; j++){
                // ppr[i]+=counts[residue.first]*1.0/config.omega*residue.second;
                int nodeid = bwd_idx.second.occur[j];
                double residual = bwd_idx.second[nodeid];
                if(nodeid == target){
                    continue;
                }
                else if(nodeid == source){
                    fhp += residual;
                }
                int occur;
                if(!sqrt_rw_counter.exist(nodeid))
                    occur = 0.0;
                else
                    occur = sqrt_rw_counter[nodeid];


                fhp += occur*1.0/(double) config.omega *residual;

            }
        }

        cout<< "final fhp: " <<fhp <<endl;
        est_ppr_file << target << " " << fhp << endl;
    }

    clock_t end = clock();
    avg_time += (end - start)/(double) CLOCKS_PER_SEC / (double) target_set.size();

    est_ppr_file.close();

}

//the version used in our papers
//re-run the experiments for pairwise query
double samba_query_pairwise_sqrt_walk_rev(int source, int target, const Graph& graph) {
    //Timer timer(BIPPR_QUERY);

   /* stringstream ss;
    ss << "estimated_fhp/" << config.graph_alias << "/pairwise/samba/" << source << ".txt";
    string outfile = ss.str();
    cout<<outfile<<endl;
    ofstream est_ppr_file(outfile);
*/
    clock_t start = clock();
    double sqrt_value = sqrt(1-config.alpha);
    double stop_prob = 1 - sqrt_value;
    //double stop_prob = config.alpha;

    //for(int i=0; i < target_set.size(); i++){
    //    int target = target_set[i];
        sqrt_rw_counter.reset_zero_values();
        //cout<<i+1 <<"-th node: " << target<<endl;
        double fhp = 0.0;

        //cout<<"rev stop probability: " << stop_prob <<endl;

       // sqrt_rw_counter.clean();
        sqrt_rw_counter.reset_zero_values();

        if (source == target) {
            fhp = 1.0;
        }
        else{
            //double new_rmax = 0.001; //10^{-6}
            INFO(config.rmax);
            //cout<< "rmax used: " << new_rmax <<endl;
            //first run the backward propagation from the target
            reverse_local_update_linear_visiting_prob_rev(source, target, graph, config.rmax);

            INFO(config.omega);

            sqrt_rw_counter.insert(source, 0);
            //rw_counter.clean();
            for (unsigned long j = 0; j < config.omega; j++) {
                //a random walk from i
                //terminates when it hits t
                int temp_node = source;
                //int step = 1;
                sqrt_rw_counter[source] += 1.0;
                //int incre = 1;
                double incre = 1.0; //to the source node itself is 1.0
                while (drand() > stop_prob)
                {
                    //step += 1;
                    //double incre = pow(sqrt(1 - config.alpha), step);
                    incre *= sqrt_value;
                    if (graph.g[temp_node].size()) {
                        int next = lrand() % graph.g[temp_node].size();
                        temp_node = graph.g[temp_node][next];

                        if(temp_node == target){
                            break;
                        }
                        else{
                            if(!sqrt_rw_counter.exist(temp_node))
                                sqrt_rw_counter.insert(temp_node, incre);
                            else
                                sqrt_rw_counter[temp_node] += incre;
                        }

                    }
                    else {
                        break;
                    }
                }
            }

            for(long j=0; j<bwd_idx.second.occur.m_num; j++){
                // ppr[i]+=counts[residue.first]*1.0/config.omega*residue.second;
                int nodeid = bwd_idx.second.occur[j];

                double residual = bwd_idx.second[nodeid];
                if(nodeid == target){
                    continue;
                }

                int occur;
                if(!sqrt_rw_counter.exist(nodeid))
                    occur = 0.0;
                else
                    occur = sqrt_rw_counter[nodeid];

                fhp += occur*1.0/(double) config.omega *residual;
            }
        }

        //cout<< "final fhp: " <<fhp <<endl;

        //est_ppr_file << target << " " << fhp << endl;
   // }

    //est_ppr_file.close();

    clock_t end = clock();
    avg_time += (end - start)/(double) CLOCKS_PER_SEC;

    return fhp;
}

void basic_fastPrune(int source, const Graph& graph, int k){
    //here we find out the top-k nodes directly by mc without the samba optimization

    unsigned long long largest_num_rw = ceil( 3 * graph.n * log(2 * graph.n) / config.epsilon / config.epsilon ); //for ground truth, epsilon should be 0.1 or 0.2 (smaller than 0.5)
    cout << "# walks: " << largest_num_rw << endl;

    sqrt_rw_counter.reset_zero_values();

    vector<double> fhps; // size of n
    vector<pair<int, double>> ordered_fhps; // size of n

    //vector<bool> candidates; //whether the node is a candidate

    unordered_set<int> candidates; //nodes whose fhp value is larger than a threshold
    int num_nodes_larger_than_lower_bounds = 0;
    vector<pair<int, double>> V_k; //the nodes which is in the answer set

    double min_delta = 1 / (double) graph.n;
    for (int i =0; i<graph.n; i++){

        if(i == source){
            fhps.push_back(1.0);
            ordered_fhps.push_back(make_pair(i, 1.0));
            //candidates.insert(i);
            num_nodes_larger_than_lower_bounds++;
            V_k.push_back(make_pair(i, 1.0));
        }
        else{
            fhps.push_back(0.0);
            ordered_fhps.push_back(make_pair(i, 0.0));
        }
        //visited.push_back(false);
        //candidates.push_back(true);
        //candidates.insert(i);
    }

    double delta = 1.0 / (double) k;
    //double sqrt_value = sqrt(1-config.alpha);
    //double stop_prob = 1 - sqrt_value;
    //cout<<"stop probability: " << stop_prob <<endl;
    int iter = 0;
    double upper_bound =1.0;
    double lower_bound = 0.0;
    //cout<<"log n:" << int(log(graph.n)) <<endl;

    double p_fail_new = config.pfail / (double) graph.n / log2(graph.n/k);
    cout<<"new fail prob: "<<p_fail_new <<endl;
    double eps_new  = config.epsilon / (double) 2.0;
    cout<<"new epsilon: " << eps_new <<endl;

    clock_t start = clock();
    unsigned long total_num_rw_only_mc =0; //since we reuse the results of previous iterations

    while(V_k.size() < k){
        cout<<"No. iter: " << iter <<endl;
        cout<< num_nodes_larger_than_lower_bounds<<endl;
        cout<<candidates.size()<<endl;

        unsigned long long num_rw = ceil(3 * log(2/ p_fail_new)/ eps_new / eps_new / delta);
        cout<<"num rw: " << num_rw <<endl;
        total_num_rw_only_mc += num_rw;
        //sqrt_rw_counter.reset_zero_values();

        //the estimation phase by mc
        if(num_nodes_larger_than_lower_bounds > k) {

            for (unsigned long j = 0; j < num_rw; j++) {
                //a random walk from i : terminates when it hits t
                int temp_node = source;

                unordered_set<int> visited_nodes;//the nodes visited in this walk
                visited_nodes.insert(source);

                while (drand() > config.alpha) {

                    if (graph.g[temp_node].size()) {
                        int next = lrand() % graph.g[temp_node].size();
                        temp_node = graph.g[temp_node][next];

                        //this node is not visited before and it is a candidate

                        if (visited_nodes.count(temp_node) == 0 && candidates.count(temp_node) > 0) {
                            if (!sqrt_rw_counter.exist(temp_node))
                                sqrt_rw_counter.insert(temp_node, 1);
                            else
                                sqrt_rw_counter[temp_node] += 1;

                            visited_nodes.insert(temp_node);
                        }
                    } else {
                        break;
                    }
                }
                visited_nodes.clear();
                //one random walk is done
            }
        }
        else {
            for (unsigned long j = 0; j < num_rw; j++) {
                //a random walk from i : terminates when it hits t
                int temp_node = source;

                unordered_set<int> visited_nodes;//the nodes visited in this walk
                visited_nodes.insert(source);

                while (drand() > config.alpha) {
                    if (graph.g[temp_node].size()) {
                        int next = lrand() % graph.g[temp_node].size();
                        temp_node = graph.g[temp_node][next];

                        //this node is not visited before and it is a candidate
                        if (visited_nodes.count(temp_node) == 0) {
                            if (!sqrt_rw_counter.exist(temp_node))
                                sqrt_rw_counter.insert(temp_node, 1);
                            else
                                sqrt_rw_counter[temp_node] += 1;

                            visited_nodes.insert(temp_node);
                        }
                    } else {
                        break;
                    }
                }
                visited_nodes.clear();
                //one random walk is done
            }
        }

        //cout<<"upper bound current " << upper_bound <<endl;
        lower_bound = (1- config.epsilon / (double) 2.0) * delta;
        upper_bound = (1+config.epsilon) * delta;
        //estimate the fhp for all nodes
        //cout<<"estimating fhps " <<endl;
        ordered_fhps.clear();
        ordered_fhps.reserve(graph.n);
        fhps.clear();
        fhps.resize(graph.n);

        if(num_nodes_larger_than_lower_bounds > k ){

            for(int i=0; i< graph.n; i++){
                if(i == source){
                    fhps[i] = 1.0;
                    ordered_fhps.push_back(make_pair(i, fhps[i])); // update the fhps based on this new samplings
                }
                else if(candidates.count(i) > 0 && (sqrt_rw_counter.exist(i))){
                    fhps[i] = sqrt_rw_counter[i] / (double) num_rw;
                    if(fhps[i] >= upper_bound){
                        V_k.push_back(make_pair(i, fhps[i]));
                        candidates.erase(i);
                    }
                    else {
                        ordered_fhps.push_back(make_pair(i, fhps[i])); // update the fhps based on this new samplings
                    }

                    //if(iter > 4){
                       if(fhps[i] <  (1-2*config.epsilon) * delta){
                          candidates.erase(i); //remove from the candidate list
                      }
                   // }
                }
            }
        }
        else{
            for(int i=0; i< graph.n; i++){
                if(i == source){
                    fhps[i] = 1.0;
                    ordered_fhps.push_back(make_pair(i, fhps[i])); // update the fhps based on this new samplings
                }
                else if(sqrt_rw_counter.exist(i)){
                    fhps[i] = sqrt_rw_counter[i] / (double) total_num_rw_only_mc;
                    ordered_fhps.push_back(make_pair(i, fhps[i])); // update the fhps based on this new samplings

                    //if(iter > 4){
                        if(fhps[i] >= lower_bound && candidates.count(i) == 0){
                            //could be considered as a candidate
                            candidates.insert(i);
                            num_nodes_larger_than_lower_bounds++;
                        }
                    //}
                }
            }
        }


        //sorting the nodes in decreasing of fhps

        cout<<"sorting fhps: " <<ordered_fhps.size() << endl;
        //size of n
        //sort(ordered_fhps.begin(), ordered_fhps.end(),[](pair<int, double> const& l, pair<int, double> const& r){return l.second > r.second;});
        //find the nodes with upper bounds and lower bounds;

        if(iter > 4 && num_nodes_larger_than_lower_bounds < k){
            break; //this is a disconnnected point
        }
        /*if(num_nodes_larger_than_lower_bounds > k && candidates.size() < k ){
            break; // the nodes in the candidate set is already enough
        }*/

        //int t_k = k- 1 - V_k.size();

        //if(ordered_fhps[t_k].second >= upper_bound){
        //    cout<<"k-th value: " << ordered_fhps[t_k].second << " " <<upper_bound <<endl;
        //    break;
        //}

        delta = delta/ (double) 2.0;
        sqrt_rw_counter.reset_zero_values();
        iter++;
    }

    clock_t end = clock();
    double total_time = (end - start) /(double) CLOCKS_PER_SEC;
    cout<<"time cost: " << total_time <<endl;
    avg_time += total_time;

    cout<<"select k nodes based on upper bounds" <<endl;
    sort(V_k.begin(), V_k.end(),[](pair<int, double> const& l, pair<int, double> const& r){return l.second > r.second;});
    stringstream ss;
    ss << "estimated_fhp/" << config.graph_alias << "/topk/basic/"<<k<<"/"<<to_str(source)<<".txt";
    string outfile = ss.str();
    ofstream topk_file(outfile);
    //put the nodes from top-k set

    int num_k = V_k.size();
    int real_k = min(num_k, k);
    for(int i=0; i<real_k; i++){
        pair<int, double> ordered_fhp = V_k[i];
        topk_file<<ordered_fhp.first << " "<< ordered_fhp.second<<endl;
    }/*
    int num_candidates = candidates.size();
    int real_k = k-V_k.size();
    real_k = min(real_k, num_candidates);
    for(int i=0; i< real_k ; i++){
        pair<int, double> ordered_fhp = ordered_fhps[i];
        topk_file<<ordered_fhp.first << " "<< ordered_fhp.second<<endl;
    }*/
    topk_file.close();
}

void mc_topk_early(int source, const Graph& graph, int k){
    //here we find out the top-k nodes directly by mc without the samba optimization

    unsigned long long largest_num_rw = ceil( 3 * graph.n * log(2 * graph.n) / config.epsilon / config.epsilon ); //for ground truth, epsilon should be 0.1 or 0.2 (smaller than 0.5)
    cout << "# walks: " << largest_num_rw << endl;

    sqrt_rw_counter.reset_zero_values();

    vector<double> fhps; // size of n
    vector<pair<int, double>> ordered_fhps; // size of n

    //vector<bool> candidates; //whether the node is a candidate

    unordered_set<int> candidates; //nodes whose fhp value is larger than a threshold
    //int num_nodes_larger_than_lower_bounds = 0;
    //vector<pair<int, double>> V_k; //the nodes which is in the answer set

    double min_delta = 1 / (double) graph.n;
    for (int i =0; i<graph.n; i++){

        if(i == source){
            fhps.push_back(1.0);
            ordered_fhps.push_back(make_pair(i, 1.0));
            //candidates.insert(i);
           // num_nodes_larger_than_lower_bounds++;
           // V_k.push_back(make_pair(i, 1.0));
        }
        else{
            fhps.push_back(0.0);
            ordered_fhps.push_back(make_pair(i, 0.0));
        }

    }

    double delta = 1.0 / (double) k;
    //double sqrt_value = sqrt(1-config.alpha);
    //double stop_prob = 1 - sqrt_value;
    //cout<<"stop probability: " << stop_prob <<endl;
    int iter = 0;
    double upper_bound =1.0;
    double lower_bound = 0.0;

    double p_fail_new = config.pfail / (double) graph.n / log2(graph.n/k);
    cout<<"new fail prob: "<<p_fail_new <<endl;
    double eps_new  = config.epsilon / (double) 2.0;
    cout<<"new epsilon: " << eps_new <<endl;

    clock_t start = clock();
    unsigned long total_num_rw_only_mc =0; //since we reuse the results of previous iterations

    while(delta >= min_delta){
        cout<<"No. iter: " << iter <<endl;
       // cout<< num_nodes_larger_than_lower_bounds<<endl;
        //cout<<candidates.size()<<endl;

        unsigned long long num_rw = ceil(3 * log(2/ p_fail_new)/ eps_new / eps_new / delta);
        cout<<"num rw: " << num_rw <<endl;
        total_num_rw_only_mc += num_rw;
        //sqrt_rw_counter.reset_zero_values();

        //the estimation phase by mc

            for (unsigned long j = 0; j < num_rw; j++) {
                //a random walk from i : terminates when it hits t
                int temp_node = source;

                unordered_set<int> visited_nodes;//the nodes visited in this walk
                visited_nodes.insert(source);

                while (drand() > config.alpha) {
                    if (graph.g[temp_node].size()) {
                        int next = lrand() % graph.g[temp_node].size();
                        temp_node = graph.g[temp_node][next];

                        //this node is not visited before and it is a candidate
                        if (visited_nodes.count(temp_node) == 0) {
                            if (!sqrt_rw_counter.exist(temp_node))
                                sqrt_rw_counter.insert(temp_node, 1);
                            else
                                sqrt_rw_counter[temp_node] += 1;

                            visited_nodes.insert(temp_node);
                        }
                    } else {
                        break;
                    }
                }
                visited_nodes.clear();
                //one random walk is done
            }


        //cout<<"upper bound current " << upper_bound <<endl;
        lower_bound = (1 - config.epsilon) * delta;
        upper_bound = (1 + config.epsilon) * delta;
        //estimate the fhp for all nodes
        //cout<<"estimating fhps " <<endl;
        ordered_fhps.clear();
        ordered_fhps.reserve(graph.n);
        fhps.clear();
        fhps.resize(graph.n);

        for(int i=0; i< graph.n; i++){
            if(i == source){
                fhps[i] = 1.0;
                ordered_fhps.push_back(make_pair(i, fhps[i])); // update the fhps based on this new samplings
            }
            else if(sqrt_rw_counter.exist(i)){
                fhps[i] = sqrt_rw_counter[i] / (double) num_rw;
                ordered_fhps.push_back(make_pair(i, fhps[i])); // update the fhps based on this new samplings
            }
            else{
                fhps[i] = 0.0;
                ordered_fhps.push_back(make_pair(i, fhps[i]));
            }
        }


        cout<<"sorting fhps: " <<ordered_fhps.size() << endl;
        //size of n
        sort(ordered_fhps.begin(), ordered_fhps.end(),[](pair<int, double> const& l, pair<int, double> const& r){return l.second > r.second;});
        //find the nodes with upper bounds and lower bounds;



        int t_k = k- 1;

        if(ordered_fhps[t_k].second >= upper_bound){
            cout<<"k-th value: " << ordered_fhps[t_k].second << " " <<upper_bound <<endl;
            break;
        }

        delta = delta/ (double) 2.0;
        sqrt_rw_counter.reset_zero_values();
        iter++;
    }

    clock_t end = clock();
    double total_time = (end - start) /(double) CLOCKS_PER_SEC;
    cout<<"time cost: " << total_time <<endl;
    avg_time += total_time;

    //cout<<"select k nodes based on upper bounds" <<endl;
    //sort(V_k.begin(), V_k.end(),[](pair<int, double> const& l, pair<int, double> const& r){return l.second > r.second;});
    stringstream ss;
    ss << "estimated_fhp/" << config.graph_alias << "/topk/mc_topk/"<<k<<"/"<<to_str(source) << "_"<<to_str(config.epsilon)<<".txt";
    string outfile = ss.str();
    ofstream topk_file(outfile);
    //put the nodes from top-k set

    for(int i=0; i< k; i++){
        pair<int, double> ordered_fhp = ordered_fhps[i];
        topk_file<<ordered_fhp.first << " "<< ordered_fhp.second<<endl;
    }
    topk_file.close();
}

void mc_topk_group_early(int source, const Graph& graph, int k){
    //here we find out the top-k groups directly by mc without the samba optimization

    unsigned long long largest_num_rw = ceil( 3 * graph.n * log(2 * graph.n) / config.epsilon / config.epsilon ); //for ground truth, epsilon should be 0.1 or 0.2 (smaller than 0.5)
    cout << "# walks: " << largest_num_rw << endl;

    rw_counter.reset_zero_values();

    vector<double> group_fhps; // size of groups list
    vector<pair<int, double>> ordered_fhps; // size of groups list

    unordered_set<int> candidates; //nodes whose fhp value is larger than a threshold
    //int num_nodes_larger_than_lower_bounds = 0;
    //vector<pair<int, double>> V_k; //the nodes which is in the answer set

    double min_delta = 1 / (double) graph.n;

    double delta = 1.0 / (double) k;
    //double sqrt_value = sqrt(1-config.alpha);
    //double stop_prob = 1 - sqrt_value;
    //cout<<"stop probability: " << stop_prob <<endl;
    int iter = 0;
    double upper_bound =1.0;
    double lower_bound = 0.0;

    double p_fail_new = config.pfail / (double) graph.n / log2(graph.n/k);
    cout<<"new fail prob: "<<p_fail_new <<endl;
    double eps_new  = config.epsilon / (double) 2.0;
    cout<<"new epsilon: " << eps_new <<endl;

    clock_t start = clock();
    unsigned long total_num_rw_only_mc =0; //since we reuse the results of previous iterations

    unsigned long long prev_num_rw = 0;
    while(delta >= min_delta){
        cout<<"No. iter: " << iter <<endl;
       // cout<< num_nodes_larger_than_lower_bounds<<endl;
        //cout<<candidates.size()<<endl;

        unsigned long long num_rw = ceil(3 * log(2/ p_fail_new)/ eps_new / eps_new / delta);
        cout<<"num rw: " << num_rw <<endl;
        total_num_rw_only_mc += num_rw;
        //rw_counter.reset_zero_values();

        //the estimation phase by mc
        for (unsigned long long j = prev_num_rw; j < num_rw; j++)
        {
            //a random walk from source
            int temp_node = source;
            //int step = 0;
            vector<int> single_walk;
            single_walk.push_back(source);
            while (drand() > config.alpha) {
                if (graph.g[temp_node].size()) {
                    int next = lrand() % graph.g[temp_node].size();
                    temp_node = graph.g[temp_node][next];
                    single_walk.push_back(temp_node);

                    /* if(temp_node == target){
                         break;

                     }
                     */

                } else {
                    break;
                }

            }

            unordered_set<int> group_hitted_set;
            for (int l = 0; l < single_walk.size(); l++)
            {
                int visited_node = single_walk[l];
                for (auto group_index: groups_inverted_list[visited_node])
                {
                    if (group_hitted_set.find(group_index) == group_hitted_set.end())
                    {
                        if (!rw_counter.exist(group_index))
                        {
                            rw_counter.insert(group_index, 1);
                        }
                        else
                        {
                            rw_counter[group_index] += 1;
                        }
                        group_hitted_set.insert(group_index);
                    }
                }
            }

        }
        prev_num_rw = num_rw;

        //cout<<"upper bound current " << upper_bound <<endl;
        lower_bound = (1 - config.epsilon) * delta;
        upper_bound = (1 + config.epsilon) * delta;
        //estimate the fhp for all nodes
        //cout<<"estimating group_fhps " <<endl;
        ordered_fhps.clear();
        ordered_fhps.reserve(groups_list.size());
        group_fhps.clear();
        group_fhps.resize(groups_list.size());

        for(int i=0; i< groups_list.size(); i++)
        {
             if(rw_counter.exist(i)){
                group_fhps[i] = rw_counter[i] / (double) num_rw;
                ordered_fhps.push_back(make_pair(i, group_fhps[i])); // update the group_fhps based on this new samplings
            }
        }

        cout<<"sorting group_fhps: " <<ordered_fhps.size() << endl;
        //size of n
        sort(ordered_fhps.begin(), ordered_fhps.end(),[](pair<int, double> const& l, pair<int, double> const& r){return l.second > r.second;});
        //find the nodes with upper bounds and lower bounds;

        int t_k = k- 1;

        if(ordered_fhps[t_k].second >= upper_bound){
            cout<<"k-th value: " << ordered_fhps[t_k].second << " " <<upper_bound <<endl;
            break;
        }

        if (delta == min_delta)
        {
            break;
        }
        delta = max(delta/ (double) 2.0, min_delta);

        //rw_counter.reset_zero_values();
        iter++;
    }

    clock_t end = clock();
    double total_time = (end - start) /(double) CLOCKS_PER_SEC;
    cout<<"time cost: " << total_time <<endl;
    avg_time += total_time;

    //cout<<"select k nodes based on upper bounds" <<endl;
    //sort(V_k.begin(), V_k.end(),[](pair<int, double> const& l, pair<int, double> const& r){return l.second > r.second;});
    string outfile = build_topk_result_file_path(k, source);;

    ofstream topk_file(outfile);
    //put the nodes from top-k set

    for(int i=0; i< k; i++){
        pair<int, double> ordered_fhp = ordered_fhps[i];
        topk_file<<ordered_fhp.first << " "<< ordered_fhp.second<<endl;
    }
    topk_file.close();
}

void samba_topk(int source, const Graph& graph, int k){

    sqrt_rw_counter.reset_zero_values();

    //call the new backward propagation

    vector<double> fhps;
    vector<pair<int, double>> ordered_fhps; //the fhps are in the decreasing order
    vector<pair<int, double>> fhp_candidates_only_mc; //candidates pruned by mc
    vector<pair<int, double>> fhp_candidates_bi; //candidates pruned by samba
    vector<pair<int,double>> V_k; //the set of nodes in the top-k list
    vector<int> candidate_list;
    //vector<bool> visited;
    int num_nodes_larger_upper_bound = 0;
    int num_nodes_larger_lower_bound = 0;
    int previous_num_larger_upper_bound = 0;
    int previous_num_larger_lower_bound = 0;
    double min_delta = 1 / (double) graph.n;
    for (int i =0; i<graph.n; i++){

        if(i == source){
            fhps.push_back(1.0);
            ordered_fhps.push_back(make_pair(i, 1.0));
        }
        else{
            fhps.push_back(0.0);
            ordered_fhps.push_back(make_pair(i, 0.0));
        }
        //visited.push_back(false);
    }

    double delta = 1.0 / (double) k /log(graph.n);

    //double sqrt_value = sqrt(1-config.alpha);
    double sqrt_value = 1.0;
    //double stop_prob = 1 - sqrt_value;
    double stop_prob = config.alpha;
    //cout<<"stop probability: " << stop_prob <<endl;
    int iter = 0;
    double upper_bound =1.0;
    double lower_bound = 0.0;
    //cout<<"log n:" << int(log(graph.n)) <<endl;

    double eps_mc = 0.5;

    double p_fail_new = config.pfail / (double) graph.n / log2(graph.n/k);
    cout<<"new fail prob: "<<p_fail_new <<endl;
    //double eps_new  = config.epsilon / (double) 2.0;
    double eps_new = eps_mc /(double) 2.0;
    cout<<"new epsilon: " << eps_new <<endl;
    clock_t mc_only_start = clock();
    unsigned long total_num_rw_only_mc =0;//since we reuse the results of previous iterations

    //while(num_nodes_larger_lower_bound < k && !(iter > 3 && previous_num_larger_lower_bound == num_nodes_larger_lower_bound) && (iter < 7)){
        //num_nodes_larger_lower_bound != num_nodes_larger_upper_bound || previous_num_larger_upper_bound != num_nodes_larger_upper_bound
    while(delta >= min_delta){
        cout<<"No. iter: " << iter <<endl;

        //cout<<"current delta: " <<delta << endl;

       unsigned long long num_rw = ceil(3 * log(2/ p_fail_new)/ eps_new / eps_new / delta);
        total_num_rw_only_mc += num_rw;
       // cout<<"num_walk: " << num_rw<<endl;

        sqrt_rw_counter.reset_zero_values();
        //the estimation phase by mc
        for (unsigned long j = 0; j < num_rw; j++) {
            //a random walk from i : terminates when it hits t
            int temp_node = source;

            unordered_set<int> visited_nodes;//the nodes visited in this walk
            visited_nodes.insert(source);

            while (drand() > config.alpha) {

                if (graph.g[temp_node].size()) {
                    int next = lrand() % graph.g[temp_node].size();
                    temp_node = graph.g[temp_node][next];

                    //this node is not visited before
                    if(visited_nodes.count(temp_node) == 0){
                        if(!sqrt_rw_counter.exist(temp_node))
                            sqrt_rw_counter.insert(temp_node, 1);
                        else
                            sqrt_rw_counter[temp_node] += 1;

                        visited_nodes.insert(temp_node);
                    }
                }
                else {
                    break;
                }
            }
            visited_nodes.clear();
            //one random walk is done
        }

        lower_bound = (1-eps_mc) * delta;

        int lower_flag = 0;
        previous_num_larger_lower_bound = num_nodes_larger_lower_bound;
        num_nodes_larger_lower_bound = 0;


        candidate_list.clear();
        candidate_list.reserve(graph.n);
        fhp_candidates_only_mc.clear();
        fhp_candidates_only_mc.reserve(graph.n);

        //estimate the fhp for all nodes
        //cout<<"estimating fhps " <<endl;
        int num_larger_delta = 1; //initially contains source node s
        for(int i=0; i< graph.n; i++){
            if(i == source){
                fhps[i] = 1.0;
                candidate_list.push_back(i);
                fhp_candidates_only_mc.push_back(make_pair(i, fhps[i]));
                //cout<<"node id: " << i << " fhp: " << fhps[i] <<endl;
                //ordered_fhps[i] = make_pair(i, 1.0);
            }
            else if(sqrt_rw_counter.exist(i)){
                fhps[i] = sqrt_rw_counter[i] / (double) num_rw;
                ordered_fhps[i]= make_pair(i, fhps[i]);
                if(fhps[i] >= delta){
                    num_larger_delta++;
                }
               // if(fhps[i] >= lower_bound){
               //     num_nodes_larger_lower_bound++;
               //     candidate_list.push_back(i);
               //     fhp_candidates_only_mc.push_back(make_pair(i, fhps[i]));
               // }
            }
        }

        if(num_larger_delta >= k){
            for(int i=0; i<graph.n; i++){
                if(fhps[i] >= lower_bound){
                   // num_nodes_larger_lower_bound++;
                    candidate_list.push_back(i);
                    fhp_candidates_only_mc.push_back(make_pair(i, fhps[i]));
                }
            }
            break;
        }

        //cout<<"# of nodes whose fhp is larger than the lower bound: " << num_nodes_larger_lower_bound <<endl;


        delta = max(delta / (double) 2.0, min_delta);
        iter++;
    }

    clock_t mc_only_end = clock();
    double mc_only_time = (mc_only_end - mc_only_start) /(double) CLOCKS_PER_SEC;
    cout<<"time cost for mc pruning: " << mc_only_time <<endl;
    //cout<<"final delta: " << delta <<endl;

    //if the number of candidate nodes is still less than $k$
    //then return the top-k node in the candidate list (ordered_fhps)

    if(candidate_list.size()<= k){
        //cout<<"select k nodes based on current version " <<endl;

        //topk_results.resize(graph.n);
        sort(ordered_fhps.begin(), ordered_fhps.end(),[](pair<int, double> const& l, pair<int, double> const& r){return l.second > r.second;});

        stringstream ss;
        ss << "estimated_fhp/" << config.graph_alias << "/topk/samba2/"<<k<<"/"<<to_str(source)<<"_" << to_str(config.epsilon) <<".txt";
        string outfile = ss.str();
        ofstream topk_file(outfile);

        for(int i=0; i<k; i++){
            pair<int, double> ordered_fhp = ordered_fhps[i];
            topk_file<<ordered_fhp.first << " "<< ordered_fhp.second<<endl;
        }

        topk_file.close();

        avg_time += mc_only_time;
        fhps.clear();
        ordered_fhps.clear();
        candidate_list.clear();
        fhp_candidates_only_mc.clear();
        return ;
    }

    //upper_bounds_fhp.clear();
    //lower_bounds_fhp.clear();

    residual_maps.clear();
    residual_maps.resize(candidate_list.size());
    visiting_prob_maps.clear();
    visiting_prob_maps.resize(candidate_list.size());
    int original_cand_size = candidate_list.size();

    node_to_order.clear();
    //unordered_set<int> candidate_nodes_hash_set;
    for(int i = 0; i< candidate_list.size(); i++){
        //candidate_nodes_hash_set.insert(candidate_list[i]);
        //lower_bounds_fhp[candidate_list[i]] = 1/(double) graph.n;
        //upper_bounds_fhp[candidate_list[i]] = 1.0;
        node_to_order[candidate_list[i]] = i;
        residual_maps[i][candidate_list[i]] = 1.0;
    }
    //double back_rmax = 1/(double) sqrt(graph.n);
    double back_rmax;
    //delta = pow(0.5, 2);
    //double stop_prob = 1- sqrt(1-config.alpha);
    int iteration_num_bi_dire = 0;

    vector<vector<int>> nonzero_residue_map;// each vector contains the targets with which the residue of v is non zero.

    for (int i =0; i<graph.n; i++){
        vector<int> nonzero_resi_v;
        nonzero_residue_map.push_back(nonzero_resi_v);
    }

    //ordered_fhps.resize(graph.n);
    //pay attention to the nodes in the candidate list
    clock_t bi_start = clock();
    double total_num_rw_bi =0;

    unsigned long long num_rw=0;
    sqrt_rw_counter.reset_zero_values();
    cout<<"candidate size: " << candidate_list.size()<<endl;

    //double eps_bi = 0.5;
    double eps_bi = config.epsilon;
    eps_new = eps_bi / 2;

    while(V_k.size()<k || (V_k.size() + candidate_list.size())< k ||delta >= min_delta){

        iteration_num_bi_dire++;
        cout<<"bi iter: " << iteration_num_bi_dire<<endl;

        back_rmax = 5* eps_new * stop_prob  *sqrt(graph.m * delta * candidate_list.size()/(double) graph.n / (double) log(2/p_fail_new)); // the new delta and p_f
        cout<<"current rmax: " << back_rmax<<endl;
        //cout<<"backward pushing for candidate nodes ... "<<endl;

        clock_t back_start_time = clock();
        for(int i=0; i<candidate_list.size(); i++){
            //do backward push for each candidate node
            int candidate_node = candidate_list[i];
            int candidate_node_order = node_to_order[candidate_node];

            if(candidate_node == source){
                nonzero_residue_map[candidate_node].push_back(candidate_node);
            }
            else{
                reverse_local_update_visiting_prob_topk_rev(source,candidate_node,back_rmax,residual_maps[candidate_node_order],graph);
                for (auto &item: residual_maps[candidate_node_order]) {
                    int node_id = item.first;
                    double resi = item.second;
                    if (node_id != candidate_node && resi > 0) {
                        nonzero_residue_map[node_id].push_back(candidate_node);
                    }
                }
            }
        }

        clock_t back_end_time = clock();
        double back_time = (back_end_time - back_start_time) / (double) CLOCKS_PER_SEC;
        cout<<"back time:" << back_time<<endl;

        double w_paramter = max(1.0, back_rmax/ delta);
        num_rw = ceil(3 * w_paramter * log(2/p_fail_new) / eps_new / eps_new);
        cout<<"current num_rw:" << num_rw<<endl;
        total_num_rw_bi += num_rw;
        sqrt_rw_counter.reset_zero_values();

        clock_t walk_start_time = clock();
        int total_num_targets_update = 0;
        for (unsigned long j = 0; j < num_rw; j++) {
            //a random walk from source which terminate with stop prob
            int temp_node = source;

            unordered_set<int> visited_nodes;
            visited_nodes.insert(source);
            //sqrt_rw_counter[source] += 1.0;
            double incre = 1.0;
            int num_targets_update = 0;
            while (drand() > stop_prob) {
                incre *= sqrt_value;
                if (graph.g[temp_node].size()) {
                    int next = lrand() % graph.g[temp_node].size();
                    temp_node = graph.g[temp_node][next];
                    visited_nodes.insert(temp_node);
                    int num_targets_nonzero_resi = nonzero_residue_map[temp_node].size();
                    for(int i = 0; i<num_targets_nonzero_resi; i++){
                            int temp_target = nonzero_residue_map[temp_node][i];
                            if(visited_nodes.count(temp_target) == 0){
                                visiting_prob_maps[node_to_order[temp_target]][temp_node] += incre;
                            }
                        num_targets_update++;
                        }
                }
                else {
                    break;
                }
            }
            visited_nodes.clear();
            //cout<<"target updates for this walk: " << num_targets_update<<endl;
            total_num_targets_update += num_targets_update;

        }

        cout<<"total number of targets updates: " << total_num_targets_update <<endl;


        //cout<<"estimating the fhp for each candidate node: " <<endl;
        fhp_candidates_bi.clear();
        for(int candidate_node: candidate_list){
            if(candidate_node == source){
                fhps[candidate_node] = 1.0;
                fhp_candidates_bi.push_back(make_pair(candidate_node, 1.0));
            }
            else{
                double fhp_s_v_bi = 0;
                int candidate_node_order = node_to_order[candidate_node];
                for (auto &item: residual_maps[candidate_node_order]) {
                    int node_id = item.first;
                    double resi = item.second;
                    //cout<<"residue: " << resi <<endl;
                    if (node_id != candidate_node && resi > 0) {
                        fhp_s_v_bi += visiting_prob_maps[candidate_node_order][node_id] * resi / (double) num_rw;
                    }
                    if (node_id == source){
                        fhp_s_v_bi += resi;
                    }
                }
                fhps[candidate_node] = fhp_s_v_bi;// + fhp_candidates_only_mc[candidate_node_order].second)/ (double) 2.0;
                fhp_candidates_bi.push_back(make_pair(candidate_node, fhps[candidate_node]));
            }
        }

        clock_t walk_end_time = clock();
        double walk_time = (walk_end_time - walk_start_time) / (double) CLOCKS_PER_SEC;
        cout<<"walking time:" << walk_time<<endl;
        //check if the k-th estimated value is larger than the upper bound or not.

        sort(fhp_candidates_bi.begin(), fhp_candidates_bi.end(), [](pair<int, double> const& l, pair<int, double> const& r){return l.second > r.second;});
        int t_k = k - 1 - V_k.size();
        upper_bound = (1 + eps_bi / (double)2.0) * delta;
        cout<<"current (1+epsilon)*delta: " << upper_bound<<endl;
        if(fhp_candidates_bi[t_k].second >= upper_bound){
            cout<<"the k-th estimated fhp value: " << fhp_candidates_bi[t_k].second <<endl;
            INFO("return the results correctly");

            stringstream ss;
            ss << "estimated_fhp/" << config.graph_alias << "/topk/samba2/"<<k<<"/"<<to_str(source)<<"_" << to_str(config.epsilon)<<".txt";
            string outfile = ss.str();
            ofstream topk_file(outfile);

            //firstly store the results in V_k
            for(int i=0; i< V_k.size(); i++){
                pair<int, double> ordered_fhp = V_k[i];
                topk_file<< ordered_fhp.first << " " << ordered_fhp.second <<endl;
            }

            int num_remaining = k - V_k.size();
            for(int i=0; i< num_remaining ; i++){
                //the nodes in V_k haven't been included yet
                pair<int, double> ordered_fhp = fhp_candidates_bi[i];
                topk_file<<ordered_fhp.first << " "<< ordered_fhp.second<<endl;
            }

            topk_file.close();

            clock_t bi_end = clock();

            double bi_time  = (bi_end - bi_start) /(double) CLOCKS_PER_SEC;
            cout<<"time cost in bi-directions: " <<bi_time<<endl;
            avg_time += mc_only_time;
            avg_time += bi_time;

            return;
        }

        //prune the nodes with fhp value larger than upper bound from the candidate set.
        //put this node into the top-k set.
        for(int i=0; i< fhp_candidates_bi.size(); i++){
            if(fhp_candidates_bi[i].second >= upper_bound){
                V_k.push_back(fhp_candidates_bi[i]);
            }
        }

        lower_bound = (1 - eps_bi) * delta;
        /*
        std::vector<int>::iterator candi_iter =  candidate_list.begin();
        int i=0;
        while(candi_iter != candidate_list.end()){
            //if(*candi_iter == fhp_candidates_bi[i].first)
             //   cout<<"matched... " <<endl;
            if(fhp_candidates_bi[i].second >= upper_bound){
                candi_iter = candidate_list.erase(candi_iter);
            }
            else if(fhp_candidates_bi[i].second < lower_bound){
                candi_iter = candidate_list.erase(candi_iter);
            }
            else{
                candi_iter++;
            }

            i++;
        }*/
        vector<int> temp;
        for (auto& elem:fhp_candidates_bi)
        {
            if( elem.second >= upper_bound || elem.second < lower_bound)
            {
                continue;
            }

            temp.push_back(elem.first);
        }

        temp.swap(candidate_list);

        cout<<"remaining candidate: " << candidate_list.size()<<endl;

        delta = max(delta / (double) 2, min_delta);

        nonzero_residue_map.clear();
        nonzero_residue_map.resize(graph.n);
        for (int i =0; i<graph.n; i++){
            vector<int> nonzero_resi_v;
            nonzero_residue_map.push_back(nonzero_resi_v);
        }
        for(int i=0; i<original_cand_size; i++){
            visiting_prob_maps[i].clear();
        }
    }
    clock_t bi_end = clock();

    double bi_time  = (bi_end - bi_start) /(double) CLOCKS_PER_SEC;
    cout<<"time cost in bi-directions: " <<bi_time<<endl;
    avg_time += mc_only_time;
    avg_time += bi_time;

    cout<<"select k nodes based on upper bounds" <<endl;

    sort(V_k.begin(), V_k.end(),[](pair<int, double> const& l, pair<int, double> const& r){return l.second > r.second;});

    stringstream ss;
    ss << "estimated_fhp/" << config.graph_alias << "/topk/samba2/"<<k<<"/"<<to_str(source)<<"_" << to_str(config.epsilon)<<".txt";
    string outfile = ss.str();
    ofstream topk_file(outfile);

    for(int i=0; i<V_k.size(); i++){
        pair<int, double> ordered_fhp = V_k[i];
        topk_file<<ordered_fhp.first << " "<< ordered_fhp.second<<endl;
    }

    topk_file.close();
}

void save_estimated_top_k(int source, vector<pair<int, double>>& fhps, int k)
{
    string outfile =build_topk_result_file_path(k, source);
    ofstream topk_file(outfile);

    for(int i=0; i<k; i++){
        pair<int, double> ordered_fhp = fhps[i];
        topk_file<<ordered_fhp.first << " "<< ordered_fhp.second<<endl;
    }

    topk_file.close();
}
inline void build_one_walk_not_restart(int start, const Graph& graph, std::vector<int>& walk)
{
    int cur = start;
    unsigned long k;

    while(true)
    {
        if (drand()< config.alpha)
        {
            return ;
        }

        if(graph.g[cur].size())
        {
            k = lrand()%graph.g[cur].size();
            cur = graph.g[cur][k];
            walk.push_back(cur);
        }
        else
        {
            return ;
        }
    }
}

void samba_second_phase_rev(int source, const Graph& graph, int k, vector<int>& candidate_list)
{
    vector<pair<int, double>> V_k;
    vector<pair<int, double>> fhp_candidates_bi; //candidates pruned by samba

    vector<int> temp_list;
    for (int i = 0; i < candidate_list.size(); i++)
    {
        if (candidate_list[i] == source)
        {
            continue;
        }
        temp_list.push_back(candidate_list[i]);
    }
    temp_list.swap(candidate_list);
    cout << "candidate list size: " << candidate_list.size() <<endl;
    V_k.push_back(make_pair(source, 1.0));

    vector<vector<int>> sample;

    residual_maps.clear();
    residual_maps.resize(candidate_list.size());

    int original_cand_size = candidate_list.size();
    double min_delta = 1 / (double) graph.n;

    double delta = config.delta_mc_stop;

    double sqrt_value = 1.0;
    double stop_prob = config.alpha;
    double upper_bound =1.0;
    double lower_bound = 0.0;

    double p_fail_new = config.pfail / (double) graph.n / log2(graph.n/k);
    double eps_new  = config.epsilon / (double) 2.0;

    node_to_order.clear();
    //unordered_set<int> candidate_nodes_hash_set;
    for(int i = 0; i< candidate_list.size(); i++){
        node_to_order[candidate_list[i]] = i;
        residual_maps[i][candidate_list[i]] = 1.0;
    }
    int iteration_num_bi_dire = 0;

    vector<int> visit_count(graph.n);
    vector<vector<int>> hit_vec(graph.n);


    //ordered_fhps.resize(graph.n);
    //pay attention to the nodes in the candidate list
    clock_t bi_start = clock();
    double total_num_rw_bi =0;

    unsigned long long num_rw=0;
    cout<<"candidate size: " << candidate_list.size()<<endl;

    double curr_rmax = 0;
    double walk_time = 0;

    vector<vector<int>> sample_vec;

    while(V_k.size()<k ){

        iteration_num_bi_dire++;
        cout<<"bi iter: " << iteration_num_bi_dire<<endl;

        curr_rmax = eps_new  * sqrt(graph.m * delta  * candidate_list.size() /(double) graph.n / (double) log(2/p_fail_new));

        cout << "curr_rmax: " << curr_rmax <<", delta:" << delta <<endl;

        clock_t back_start_time = clock();

        for(int i=0; i<candidate_list.size(); i++){
            //do backward push for each candidate node
            int candidate_node = candidate_list[i];
            int candidate_node_order = node_to_order[candidate_node];
            reverse_local_update_visiting_prob_topk_rev(source, candidate_node, curr_rmax, residual_maps[candidate_node_order], graph);
        }
        clock_t back_end_time = clock();
        double back_time = (back_end_time - back_start_time) / (double) CLOCKS_PER_SEC;
        cout<<"back time:" << back_time<<endl;
        backpush_time += back_time;

        double w_paramter = max(1.0, curr_rmax/ delta);
        num_rw = ceil(3 * w_paramter * log(2/p_fail_new) / eps_new / eps_new) ;
        cout<<"current num_rw:" << num_rw<<endl;
        total_num_rw_bi += num_rw;


        clock_t walk_start_time = clock();
        int prev_size = sample_vec.size();
        for (int i = prev_size; i< num_rw; i++)
        {
            std::vector<int> one_walk;
            build_one_walk_not_restart(source, graph, one_walk);
            sample_vec.push_back(one_walk);
        }

        for (int sample_id = prev_size; sample_id < num_rw; sample_id++)
        {
            const auto& walk = sample_vec[sample_id];

            for (int j = 0; j < walk.size(); j++)
            {
                int node = walk[j];
                visit_count[walk[j]]++;

                //顺序不能变，短路运算
                if (0 == hit_vec[node].size() || hit_vec[node].back() != sample_id)
                {
                    hit_vec[node].push_back(sample_id);
                }
            }
        }

        fhp_candidates_bi.clear();

        const double incr = 1.0/num_rw;
        for (int candidate: candidate_list)
        {
            int candidate_node_order = node_to_order[candidate];
            auto &rmap = residual_maps[candidate_node_order];
            double fhp = 0.0;
            for (auto elem: rmap)
            {
                fhp += 1.0*visit_count[elem.first] * elem.second/num_rw;
            }

            fhp += rmap[source];

            for (int index: hit_vec[candidate])
            {
                const auto &walk = sample_vec[index];
                bool found = false;
                for (int node_id: walk)
                {
                    if (!found && node_id == candidate)
                    {
                        found = true;
                        continue;
                    }
                    else
                    {
                        fhp -= rmap[node_id]*incr;
                    }
                }
            }

            fhp_candidates_bi.push_back(make_pair(candidate, fhp));
        }



        clock_t walk_end_time = clock();
        walk_time += (walk_end_time - walk_start_time) / (double) CLOCKS_PER_SEC;
        cout<<"walking time:" << walk_time<<endl;

        sort(fhp_candidates_bi.begin(), fhp_candidates_bi.end(), [](pair<int, double> const& l, pair<int, double> const& r){return l.second > r.second;});

        int t_k = k - 1 - V_k.size();
        upper_bound = (1+config.epsilon) * delta;
        cout<<"current (1+epsilon)*delta: " << upper_bound<<endl;

        if(fhp_candidates_bi[t_k].second >= upper_bound){

            int num_remaining = k - V_k.size();
            for (int i = 0; i < num_remaining; i++)
            {
                V_k.push_back(fhp_candidates_bi[i]);
            }

            save_estimated_top_k(source, V_k, k);

            clock_t bi_end = clock();

            double bi_time  = (bi_end - bi_start) /(double) CLOCKS_PER_SEC;
            cout<<"time cost in bi-directions: " <<bi_time<<endl;
            avg_time += bi_time;
            second_phase_time += bi_time;

            return;
        }

        //prune the nodes with fhp value larger than upper bound from the candidate set.
        //put this node into the top-k set.
        for(int i=0; i< fhp_candidates_bi.size(); i++){
            if(fhp_candidates_bi[i].second >= upper_bound){
                V_k.push_back(fhp_candidates_bi[i]);
            }
        }

        lower_bound = (1 - config.epsilon) * delta;
        vector<int> temp;
        for (auto& elem:fhp_candidates_bi)
        {
            if( elem.second >= upper_bound || elem.second < lower_bound)
            {
                continue;
            }

            temp.push_back(elem.first);
        }

        temp.swap(candidate_list);

        cout<<"remaining candidate: " << candidate_list.size()<<endl;

        delta = max(delta / 2.0, min_delta);
    }
    clock_t bi_end = clock();

    double bi_time  = (bi_end - bi_start) /(double) CLOCKS_PER_SEC;
    cout<<"time cost in bi-directions: " <<bi_time<<endl;
    avg_time += bi_time;
    second_phase_time += bi_time;

    cout<<"select k nodes based on upper bounds" <<endl;

    save_estimated_top_k(source, V_k, k);
}

inline void build_one_walk_sqrt_root(int start, const Graph& graph, std::vector<int>& walk, const double stop_prob)
{
    int cur = start;
    unsigned long k;

    while(true)
    {
        if (drand()< stop_prob)
        {
            return ;
        }

        if(graph.g[cur].size())
        {
            k = lrand()%graph.g[cur].size();
            cur = graph.g[cur][k];
            walk.push_back(cur);
        }
        else
        {
            return ;
        }
    }
}

void samba_second_phase_sqrt_walk(int source, const Graph& graph, int k, vector<int>& candidate_list)
{
    vector<pair<int, double>> V_k;
    vector<pair<int, double>> fhp_candidates_bi; //candidates pruned by samba


    vector<int> temp_list;
    for (int i = 0; i < candidate_list.size(); i++)
    {
        if (candidate_list[i] == source)
        {
            continue;
        }
        temp_list.push_back(candidate_list[i]);
    }
    temp_list.swap(candidate_list);
    cout << "candidate list size: " << candidate_list.size() <<endl;
    V_k.push_back(make_pair(source, 1.0));

    vector<vector<int>> sample;

    residual_maps.clear();
    residual_maps.resize(candidate_list.size());

    int original_cand_size = candidate_list.size();
    double min_delta = 1 / (double) graph.n;

    double delta = config.delta_mc_stop;

    const double sqrt_value = sqrt(1.0 - config.alpha);
    double stop_prob = 1.0-sqrt_value;
    double upper_bound =1.0;
    double lower_bound = 0.0;

    double p_fail_new = config.pfail / (double) graph.n / log2(graph.n/k);
    double eps_new  = config.epsilon / (double) 2.0;

    node_to_order.clear();
    //unordered_set<int> candidate_nodes_hash_set;
    for(int i = 0; i< candidate_list.size(); i++){
        node_to_order[candidate_list[i]] = i;
        residual_maps[i][candidate_list[i]] = 1.0;
    }
    int iteration_num_bi_dire = 0;

    vector<double> visit_count(graph.n);
    vector<vector<int>> hit_vec(graph.n);


    //ordered_fhps.resize(graph.n);
    //pay attention to the nodes in the candidate list
    clock_t bi_start = clock();
    double total_num_rw_bi =0;

    unsigned long long num_rw=0;
    cout<<"candidate size: " << candidate_list.size()<<endl;

    double curr_rmax = 0;
    double walk_time = 0;

    vector<vector<int>> sample_vec;

    while(V_k.size()<k ){

        iteration_num_bi_dire++;
        cout<<"bi iter: " << iteration_num_bi_dire<<endl;

        curr_rmax = eps_new  * sqrt(stop_prob / config.alpha *graph.m * delta  * candidate_list.size() /(double) graph.n / (double) log(2/p_fail_new));

        cout << "curr_rmax: " << curr_rmax <<", delta:" << delta <<endl;

        clock_t back_start_time = clock();

        for(int i=0; i<candidate_list.size(); i++){
            //do backward push for each candidate node
            int candidate_node = candidate_list[i];
            int candidate_node_order = node_to_order[candidate_node];
            reverse_local_update_visiting_prob_topk_rev(source, candidate_node, curr_rmax, residual_maps[candidate_node_order], graph);
        }
        clock_t back_end_time = clock();
        double back_time = (back_end_time - back_start_time) / (double) CLOCKS_PER_SEC;
        cout<<"back time:" << back_time<<endl;
        backpush_time += back_time;

        double w_paramter = max(1.0, curr_rmax/ delta);
        num_rw = ceil(3 * w_paramter * log(2/p_fail_new) / eps_new / eps_new) ;
        cout<<"current num_rw:" << num_rw<<endl;
        total_num_rw_bi += num_rw;


        clock_t walk_start_time = clock();
        int prev_size = sample_vec.size();
        for (int i = prev_size; i< num_rw; i++)
        {
            std::vector<int> one_walk;
            build_one_walk_sqrt_root(source, graph, one_walk, stop_prob);
            sample_vec.push_back(one_walk);
        }

        long long hit_count = 0;
        for (int sample_id = prev_size; sample_id < num_rw; sample_id++)
        {
            double incr = 1.0;
            const auto& walk = sample_vec[sample_id];

            for (int j = 0; j < walk.size(); j++)
            {
                incr *= sqrt_value;
                int node = walk[j];
                visit_count[node] += incr;

                //顺序不能变，短路运算
                if (0 == hit_vec[node].size() || hit_vec[node].back() != sample_id)
                {
                    hit_vec[node].push_back(sample_id);
                }
            }
        }

        fhp_candidates_bi.clear();

        for (int candidate: candidate_list)
        {
            int candidate_node_order = node_to_order[candidate];
            auto &rmap = residual_maps[candidate_node_order];
            double fhp = 0.0;
            for (auto elem: rmap)
            {
                fhp += 1.0*visit_count[elem.first] * elem.second/num_rw;
            }

            fhp += rmap[source];
            hit_count += hit_vec[candidate].size();

            for (int index: hit_vec[candidate])
            {
                const auto &walk = sample_vec[index];
                bool found = false;

                double incr = 1.0/num_rw;
                for (int node_id: walk)
                {
                    incr *= sqrt_value;
                    if (!found && node_id == candidate)
                    {
                        found = true;
                        continue;
                    }
                    else
                    {
                        fhp -= rmap[node_id]*incr;
                    }
                }
            }

            fhp_candidates_bi.push_back(make_pair(candidate, fhp));
        }



        clock_t walk_end_time = clock();
        walk_time += (walk_end_time - walk_start_time) / (double) CLOCKS_PER_SEC;
        cout<<"walking time:" << walk_time<<endl;

        sort(fhp_candidates_bi.begin(), fhp_candidates_bi.end(), [](pair<int, double> const& l, pair<int, double> const& r){return l.second > r.second;});

        int t_k = k - 1 - V_k.size();
        upper_bound = (1+config.epsilon) * delta;
        cout<<"current (1+epsilon)*delta: " << upper_bound<<endl;

        if(fhp_candidates_bi[t_k].second >= upper_bound){

            int num_remaining = k - V_k.size();
            for (int i = 0; i < num_remaining; i++)
            {
                V_k.push_back(fhp_candidates_bi[i]);
            }

            save_estimated_top_k(source, V_k, k);

            clock_t bi_end = clock();

            double bi_time  = (bi_end - bi_start) /(double) CLOCKS_PER_SEC;
            cout<<"time cost in bi-directions: " <<bi_time<<endl;
            avg_time += bi_time;
            second_phase_time += bi_time;

            return;
        }

        //prune the nodes with fhp value larger than upper bound from the candidate set.
        //put this node into the top-k set.
        for(int i=0; i< fhp_candidates_bi.size(); i++){
            if(fhp_candidates_bi[i].second >= upper_bound){
                V_k.push_back(fhp_candidates_bi[i]);
            }
        }

        lower_bound = (1 - config.epsilon) * delta;
        vector<int> temp;
        for (auto& elem:fhp_candidates_bi)
        {
            if(  elem.second < lower_bound)
            {
                continue;
            }

            temp.push_back(elem.first);
        }

        temp.swap(candidate_list);

        cout<<"remaining candidate: " << candidate_list.size()<<endl;

        delta = max(delta / 2.0, min_delta);
    }
    clock_t bi_end = clock();

    double bi_time  = (bi_end - bi_start) /(double) CLOCKS_PER_SEC;
    cout<<"time cost in bi-directions: " <<bi_time<<endl;
    avg_time += bi_time;
    second_phase_time += bi_time;

    cout<<"select k nodes based on upper bounds" <<endl;

    save_estimated_top_k(source, V_k, k);
}

void samba_first_phase(int source, const Graph& graph, int k, vector<int>& candidate_list)
{
    sqrt_rw_counter.clean();

    vector<double> fhps;
    vector<pair<int, double>> ordered_fhps; //the fhps are in the decreasing order
    vector<pair<int,double>> V_k; //the set of nodes in the top-k list

    int num_nodes_larger_than_delta = 0;
    double min_delta = 1 / (double) graph.n;

    for (int i =0; i<graph.n; i++){
        fhps.push_back(0.0);
        ordered_fhps.push_back(make_pair(i, 0.0));
    }
    fhps[source]= 1.0;
    ordered_fhps[source].second = 1.0;

    double delta = 1.0 / (double) k ;

    double sqrt_value = 1.0;
    double stop_prob = config.alpha;
    int iter = 0;
    double upper_bound =1.0;
    double lower_bound = 0.0;

    double p_fail_new = config.pfail / (double) graph.n / log2(graph.n/k);
    cout<<"new fail prob: "<<p_fail_new <<endl;
    double eps_new  = config.epsilon_prune / (double) 2.0;
    cout<<"new epsilon_prune: " << eps_new <<endl;
    clock_t mc_only_start = clock();
    unsigned long total_num_rw_only_mc =0;//since we reuse the results of previous iterations

    while(num_nodes_larger_than_delta < k ){


        cout<<"No. iter: " << iter << ", delta: " << delta << endl;

        unsigned long long num_rw = ceil(3 * log(2/ p_fail_new)/ eps_new / eps_new / delta);
        total_num_rw_only_mc += num_rw;

        sqrt_rw_counter.clean();
        //the estimation phase by mc

        sqrt_rw_counter.insert(source, 1);
        for (unsigned long j = 0; j < num_rw; j++) {
            int temp_node = source;

            unordered_set<int> visited_nodes;//the nodes visited in this walk
            visited_nodes.insert(source);

            sqrt_rw_counter[source] += 1.0;
            while (drand() > config.alpha) {

                if (graph.g[temp_node].size()) {
                    int next = lrand() % graph.g[temp_node].size();
                    temp_node = graph.g[temp_node][next];

                    //this node is not visited before
                    if(visited_nodes.count(temp_node) == 0)
                    {
                        if(!sqrt_rw_counter.exist(temp_node))
                            sqrt_rw_counter.insert(temp_node, 1);
                        else
                            sqrt_rw_counter[temp_node] += 1;

                        visited_nodes.insert(temp_node);
                    }
                }
                else {
                    break;
                }
            }
            visited_nodes.clear();
            //one random walk is done
        }

        lower_bound =  (1.0-config.epsilon_prune)*delta;
        num_nodes_larger_than_delta = 0;

        candidate_list.clear();
        candidate_list.reserve(graph.n);

        vector<int> temp;
        long long exist_count = 0;
        for(int i=0; i< graph.n; i++)
        {
            if(sqrt_rw_counter.exist(i)){
                exist_count++;
                fhps[i] = sqrt_rw_counter[i] / (double) num_rw;
                ordered_fhps[i].second = fhps[i];
                if(fhps[i] >= delta){
                    num_nodes_larger_than_delta++;
                    candidate_list.push_back(i);
                }
                else if(fhps[i] >= lower_bound)
                {
                    temp.push_back(i);
                }

            }
        }



        if (num_nodes_larger_than_delta >= k)
        {
            cout << "non zero value: " << (double)exist_count/graph.n*100 << "%"<<endl;
            cout << "num of node larger than delta： " << candidate_list.size() <<endl;
            for (int node: temp)
            {
                candidate_list.push_back(node);
            }
        }

        delta = max(delta / (double) 2.0, min_delta);
        iter++;
    }

    clock_t mc_only_end = clock();
    double mc_only_time = (mc_only_end - mc_only_start) /(double) CLOCKS_PER_SEC;
    cout<<"time cost for mc pruning: " << mc_only_time <<endl;
    avg_time += mc_only_time;
    first_phase_time += mc_only_time;

    // 保存当前的delta，在second phase使用
    config.delta_mc_stop = delta;
}

//fast prune
void fast_prune(int source, const Graph& graph, int k){

    vector<int> candidate_list;
    samba_first_phase(source, graph, k, candidate_list);

    if (candidate_list.size() < k)
    {
        cout << "it is disconnnected node" << endl;
        return ;
    }

    if(config.sqrt_walk){
        samba_second_phase_sqrt_walk(source, graph, k, candidate_list);
    }else{
        samba_second_phase_rev(source, graph, k, candidate_list);
    }

    return ;
}
void build_group_inv_list(const vector<vector<int>>& groups, int graph_size)
{
    groups_inverted_list.resize(graph_size);
    for (int index= 0; index < groups.size(); index++)
    {
        for (int node: groups[index])
        {
            if (groups_inverted_list[node].size() == 0)
            {
                groups_inverted_list[node].push_back(index);
            }
            else
            {
                auto last_element_index = groups_inverted_list[node].size()-1;
                if (groups_inverted_list[node][last_element_index] != index)
                {
                    groups_inverted_list[node].push_back(index);
                }
                else
                {
                    cout << "same node i one group" <<endl;
                }
            }
        }
    }
}

void convert_candidate_list_to_inverted_map(const vector<int>& candidate_list)
{
    unordered_map<int, vector<int>> temp_inverted;

    for (int index = 0; index < candidate_list.size(); index++)
    {
        int group_index = candidate_list[index];
        if (group_index < 0)
        {
            continue;
        }

        for (auto member: groups_list[group_index])
        {
 //           if (temp_inverted.find(member)!=temp)
            vector<int>& inverted_vec = temp_inverted[member];

            int inverted_size = inverted_vec.size();
            if ( inverted_size == 0 || temp_inverted[member][inverted_size-1] != group_index)
            {
                inverted_vec.push_back(group_index);
            }
        }
    }

    temp_inverted.swap(candidate_groups_inverted_map);
}

void samba_first_phase_for_topk_group(int source, const Graph& graph, int k, vector<int>& candidate_list, vector<pair<int, double>>& ordered_fhps_group)
{
    sqrt_rw_counter.clean();

    vector<double> fhps_group;//the fhps are in the decreasing order

    int num_nodes_larger_than_delta = 0;
    double min_delta = 1 / (double) graph.n;

    // for (int i =0; i<groups_list.size(); i++){
    //     fhps_group.push_back(0.0);
    // }
    //fhps_group[source]= 1.0;

    //ordered_fhps_group[source].second = 1.0;

    double delta = 1.0 / (double) k ;

    double sqrt_value = 1.0;
    double stop_prob = config.alpha;
    int iter = 0;
    double upper_bound =1.0;
    double lower_bound = 0.0;

    int  max_iteration = ceil(log2(graph.n/k));

    double min_epsilon = config.epsilon/2.0;
    double p_fail_new = config.pfail / (double) graph.n / (double)max_iteration;
    cout<<"new fail prob: "<<p_fail_new <<endl;
    double eps_new  = config.epsilon_prune / (double) 2.0;
    cout<<"new epsilon_prune: " << eps_new <<endl;
    clock_t mc_only_start = clock();
    unsigned long total_num_rw_only_mc =0;//since we reuse the results of previous iterations
    int reach_min_delta = 0;

    sqrt_rw_counter.clean();
    ordered_fhps_group.clear();
    //the estimation phase by mc

    for (auto index: groups_inverted_list[source])
    {
        sqrt_rw_counter.insert(index, 0);
    }

    unsigned long prev_num_rw = 0;
    while(iter < max_iteration){


        cout<<"No. iter: " << iter << ", delta: " << delta << ", prune epsilon: " << eps_new <<endl;

        unsigned long long num_rw = ceil(3 * log(2/ p_fail_new)/ eps_new / eps_new / delta);
        total_num_rw_only_mc += num_rw;


        for (unsigned long j = prev_num_rw; j < num_rw; j++) {
            int temp_node = source;

            unordered_set<int> visited_groups;//the nodes visited in this walk
            for (auto index: groups_inverted_list[source])
            {
                visited_groups.insert(index);
                sqrt_rw_counter[index] += 1.0;
            }

            while (drand() > config.alpha) {

                if (graph.g[temp_node].size()) {
                    int next = lrand() % graph.g[temp_node].size();
                    temp_node = graph.g[temp_node][next];

                    for(auto index: groups_inverted_list[temp_node])
                    {
                       //this node is not visited before
                        if(visited_groups.count(index) == 0)
                        {
                            if(!sqrt_rw_counter.exist(index))
                                sqrt_rw_counter.insert(index, 1);
                            else
                                sqrt_rw_counter[index] += 1;

                            visited_groups.insert(index);
                        }
                    }

                }
                else {
                    break;
                }
            }
            visited_groups.clear();
            //one random walk is done
        }
        prev_num_rw = num_rw;

        lower_bound =  (1.0-config.epsilon_prune)*delta;
        num_nodes_larger_than_delta = 0;

        candidate_list.clear();
        ordered_fhps_group.clear();

        vector<int> temp_list;
        long long exist_count = 0;
        double group_fhps = 0.0;
        for(int i=0; i< groups_list.size(); i++)
        {
            if(sqrt_rw_counter.exist(i)){
                exist_count++;
                group_fhps = sqrt_rw_counter[i] / (double) num_rw;
                if(group_fhps >= delta){
                    num_nodes_larger_than_delta++;
                    candidate_list.push_back(i);
                    ordered_fhps_group.push_back(make_pair(i, group_fhps));

                }
                else if(group_fhps>= lower_bound)
                {
                    candidate_list.push_back(i);
                    ordered_fhps_group.push_back(make_pair(i, group_fhps));
                }

            }
        }


        if (eps_new == min_epsilon)
        {
            double upper_bound = (1+config.epsilon)*delta;
            sort(ordered_fhps_group.begin(), ordered_fhps_group.end(), [](pair<int, double> const& l, pair<int, double> const& r){return l.second > r.second;});
            if (ordered_fhps_group[k-1].second>upper_bound)
            {
                cout << "found topk now, the last fhp：" << ordered_fhps_group[k-1].first << ", " <<ordered_fhps_group[k-1].second <<endl;
                candidate_list.clear();

                ordered_fhps_group.resize(k);
                for (int i =0; i<k; i++)
                {
                    candidate_list.push_back(ordered_fhps_group[i].first);
                }
                break;
            }
            else if (delta > min_delta)
            {
                delta = delta / (double) 2.0;
                continue;
            }
            else
            {
                break;
            }
        }

        if (num_nodes_larger_than_delta >= k)
        {
            //cout << "non zero value: " << (double)exist_count/groups_list.size()*100 << "%"<<endl;
            cout << "num of node larger than delta： " << num_nodes_larger_than_delta;
            cout << ", total candidate number: " << candidate_list.size() <<endl;

            double magic_num = 0.5;
            double threshold = k *(1.0/config.epsilon)/magic_num;
            cout << "threshold: " << threshold << endl;
            if (candidate_list.size() > threshold)
            {
                eps_new = max(min_epsilon, eps_new/sqrt(2.0));
                iter++;
                continue;
            }
            else
            {
                break;
            }

            // sort(ordered_fhps_group.begin(), ordered_fhps_group.end(), [](pair<int, double> const& l, pair<int, double> const& r){return l.second > r.second;});

            // double new_delta = ordered_fhps_group[k].second;
            // double new_lowerbound = new_delta * (1.0-config.epsilon_prune);

            // candidate_list.clear();

            // for (auto& node: ordered_fhps_group)
            // {
            //     if (node.second < new_lowerbound)
            //     {
            //         break;
            //     }

            //     candidate_list.push_back(node.first);
            // }
            // cout << "new delta: " << new_delta << ", num of node larger than new delta： " << candidate_list.size() <<endl;
            // delta = new_delta;
        }
        else
        {
            cout << "larger than delta: " << num_nodes_larger_than_delta;
            cout << ", larger than lower bound: " << candidate_list.size() << endl;
        }


        delta = delta / (double) 2.0;
        if (delta <= min_delta)
        {
            delta = min_delta;
            reach_min_delta++;
        }
        iter++;
    }

    clock_t mc_only_end = clock();
    double mc_only_time = (mc_only_end - mc_only_start) /(double) CLOCKS_PER_SEC;
    cout<<"time cost for mc pruning: " << mc_only_time <<endl;
    avg_time += mc_only_time;
    first_phase_time += mc_only_time;

    // 保存当前的delta，在second phase使用
    config.delta_mc_stop = delta;
}


void samba_second_phase_sqrt_walk_for_topk_group(int source, const Graph& graph, int k, vector<int>& candidate_list)
{
    vector<pair<int, double>> V_k;
    vector<pair<int, double>> fhp_candidates_bi; //candidates pruned by samba

    cout << "candidate list size: " << candidate_list.size() <<endl;

    auto &source_group_vec = groups_inverted_list[source];
    if (source_group_vec.size()!=0)
    {
        unordered_set<int> source_group_set(source_group_vec.begin(), source_group_vec.end());
        int back_index = 0;
        for (int front_index = 0; front_index < candidate_list.size(); front_index++)
        {
            if (source_group_set.find(candidate_list[front_index]) != source_group_set.end())
            {
                V_k.push_back(make_pair(candidate_list[front_index], 1.0));
                continue;
            }

            if (back_index != front_index)
            {
                candidate_list[back_index] = candidate_list[front_index];

            }

            back_index++;
        }
        candidate_list.resize(back_index);
    }
    cout << "new candidate list size: " << candidate_list.size() << ", V_k size: " << V_k.size()<<endl;
    convert_candidate_list_to_inverted_map(candidate_list);

    vector<vector<int>> sample;
    vector<double> reserve_vec;

    int original_cand_size = candidate_list.size();
    double min_delta = 1 / (double) graph.n;

    double delta = config.delta_mc_stop;

    const double sqrt_value = sqrt(1.0 - config.alpha);
    double stop_prob = 1.0-sqrt_value;
    //double stop_prob = config.alpha;
    double upper_bound =1.0;
    double lower_bound = 0.0;

    double p_fail_new = config.pfail / (double) graph.n / log2(graph.n/k);
    double eps_new  = config.epsilon / (double) 2.0;

    node_to_order.clear();
    //unordered_set<int> candidate_nodes_hash_set;
    for(int i = 0; i< candidate_list.size(); i++){
        int group_index = candidate_list[i];

        node_to_order[group_index] = i;
    }
    int iteration_num_bi_dire = 0;

    vector<double> visit_count(graph.n);
    vector<vector<int>> group_hit_vec(groups_list.size());

    //ordered_fhps_group.resize(graph.n);
    //pay attention to the nodes in the candidate list
    clock_t bi_start = clock();
    double total_num_rw_bi =0;

    unsigned long long num_rw=0;
    cout<<"candidate size: " << candidate_list.size()<<endl;

    double curr_rmax = 0;
    double walk_time = 0;

    vector<vector<int>> sample_vec;
    int group_size = groups_list[0].size();

    residual_maps.clear();
    residual_maps.resize(candidate_list.size());
    reserve_vec.clear();
    reserve_vec.resize(candidate_list.size());
    for (int index =0; index < candidate_list.size(); index++)
    {
        auto &group = groups_list[candidate_list[index]];
        for (auto node:group)
        {
            residual_maps[index][node]=1.0;
        }
    }


    int candidate_num = candidate_list.size();
    while(V_k.size()<k )
    {

        iteration_num_bi_dire++;
        cout<<"bi iter: " << iteration_num_bi_dire<<endl;
         curr_rmax = eps_new  * sqrt(stop_prob / config.alpha *graph.m * delta  * group_size  * candidate_num/(double) graph.n / (double) log(2/p_fail_new));
        //curr_rmax = eps_new  * group_size * sqrt(stop_prob / config.alpha *graph.m * delta  * candidate_num/(double) graph.n / (double) log(2/p_fail_new));

        //curr_rmax = 0.25;
        cout << "curr_rmax: " << curr_rmax <<", delta:" << delta <<endl;
        if (curr_rmax > 1)
        {
            cout << "curr max is larger than 1, " << curr_rmax << endl;
            curr_rmax = min(curr_rmax/sqrt(group_size), 0.5);
        }

        cout << "curr_rmax: " << curr_rmax <<", delta:" << delta <<endl;

        clock_t back_start_time = clock();

        for(int i=0; i<candidate_list.size(); i++)
        {
            //do backward push for each candidate node
            int group_index = candidate_list[i];

            if (group_index < 0)
            {
                continue;
            }

            int candidate_list_index = node_to_order[group_index];
            // cout << source << ", " << reserve_vec[candidate_list_index];
            backward_push_for_topk_group(source, groups_list[group_index], graph, curr_rmax,residual_maps[candidate_list_index], reserve_vec[candidate_list_index]);
            // cout << ", after: " << reserve_vec[candidate_list_index] << endl;
       }

        clock_t back_end_time = clock();
        double back_time = (back_end_time - back_start_time) / (double) CLOCKS_PER_SEC;
        cout<<"back time:" << back_time<<endl;
        backpush_time += back_time;

        double w_paramter = curr_rmax/ delta;
        unsigned long long new_num_rw = ceil(3 * w_paramter * log(2/p_fail_new) / eps_new / eps_new);
        num_rw = max(num_rw, new_num_rw);
        cout<<"current num_rw:" << num_rw<<endl;
        total_num_rw_bi += num_rw;


        clock_t walk_start_time = clock();
        int prev_size = sample_vec.size();
        for (int i = prev_size; i< num_rw; i++)
        {
            std::vector<int> one_walk;
            build_one_walk_sqrt_root(source, graph, one_walk, stop_prob);
            sample_vec.push_back(one_walk);
        }

        long long hit_count = 0;
        for (int sample_id = prev_size; sample_id < num_rw; sample_id++)
        {
            double incr = 1.0;
            const auto& walk = sample_vec[sample_id];

            for (int j = 0; j < walk.size(); j++)
            {
                incr *= sqrt_value;
                int node = walk[j];
                visit_count[node] += incr;

                for (auto& group_index: candidate_groups_inverted_map[node])
                {
                    auto& group_hit = group_hit_vec[group_index];
                    int last_node = group_hit.size();
                    //顺序不能变，短路运算
                    if (0 == last_node || group_hit[last_node - 1] != sample_id)
                    {
                        group_hit.push_back(sample_id);
                    }
                }
            }
        }

        fhp_candidates_bi.clear();

        for (int group_index: candidate_list)
        {
            if (group_index < 0)
            {
                continue;
            }

            int candidate_list_order = node_to_order[group_index];
            auto &rmap = residual_maps[candidate_list_order];
            double fhp = 0.0;
            for (auto& elem: rmap)
            {
                fhp += 1.0 * visit_count[elem.first] * elem.second/num_rw;
            }

            /* random walk does not contain the source */
            fhp += rmap[source];
            fhp += reserve_vec[candidate_list_order];


            unordered_set<int> t_set(groups_list[group_index].begin(), groups_list[group_index].end());
            for (int index: group_hit_vec[group_index])
            {
                const auto &walk = sample_vec[index];
                bool found = false;

                double incr = 1.0/num_rw;
                for (int node_id: walk)
                {
                    incr *= sqrt_value;
                    if (!found && t_set.find(node_id) != t_set.end())
                    {
                        found = true;
                        continue;
                    }
                    else
                    {
                        fhp -= rmap[node_id]*incr;
                    }
                }
            }

            fhp_candidates_bi.push_back(make_pair(group_index, fhp));
        }

        clock_t walk_end_time = clock();
        walk_time += (walk_end_time - walk_start_time) / (double) CLOCKS_PER_SEC;
        cout<<"walking time:" << walk_time<<endl;

        cout<< "fhp list size： " << fhp_candidates_bi.size() << endl;
        sort(fhp_candidates_bi.begin(), fhp_candidates_bi.end(), [](pair<int, double> const& l, pair<int, double> const& r){return l.second > r.second;});

        int t_k = k - 1 - V_k.size();
        upper_bound = (1+config.epsilon) * delta;
        cout<<"current (1+epsilon)*delta: " << upper_bound<<endl;

        if(fhp_candidates_bi[t_k].second >= upper_bound){

            int num_remaining = k - V_k.size();
            for (int i = 0; i < num_remaining; i++)
            {
                V_k.push_back(fhp_candidates_bi[i]);
            }
            print_test_time();

            save_estimated_top_k(source, V_k, k);

            clock_t bi_end = clock();

            double bi_time  = (bi_end - bi_start) /(double) CLOCKS_PER_SEC;
            cout<<"time cost in bi-directions: " <<bi_time << ", after walking end time: " << (bi_end - walk_end_time) /(double) CLOCKS_PER_SEC<<endl;
            //avg_time += bi_time;
            second_phase_time += bi_time;

            return;
        }

        lower_bound = (1 - config.epsilon) * delta;
        int new_count = 0;
        //prune the nodes with fhp value larger than upper bound from the candidate set.
        //put this node into the top-k set.
        for(int i=0; i< fhp_candidates_bi.size(); i++)
        {
            int node = fhp_candidates_bi[i].first;
            double fhp = fhp_candidates_bi[i].second;

            int candidate_list_index = node_to_order[node];

            if(fhp >= upper_bound)
            {

                candidate_list[candidate_list_index] = -1;
                V_k.push_back(fhp_candidates_bi[i]);
                continue;
            }
            else if(fhp < lower_bound)
            {
                //cout << "remove: " << node << ": " << fhp << endl;
                candidate_list[candidate_list_index] = -1;
                continue;
            }
            else
            {
                //cout << "reserve: " << node << ": " << fhp << endl;
                new_count++;
            }

        }

        if (candidate_num > 2*new_count)
        {
            convert_candidate_list_to_inverted_map(candidate_list);
        }

        candidate_num = new_count;
        cout<<"remaining candidate: " << new_count << ", current Vk: " << V_k.size() <<endl;

        delta = max(delta / 2.0, min_delta);
    }
    clock_t bi_end = clock();

    double bi_time  = (bi_end - bi_start) /(double) CLOCKS_PER_SEC;
    cout<<"time cost in bi-directions: " <<bi_time<<endl;
    //avg_time += bi_time;
    second_phase_time += bi_time;

    cout<<"select k nodes based on upper bounds" <<endl;

    save_estimated_top_k(source, V_k, k);
}
//fast prune
void fast_prune_group(int source, const Graph& graph, int k){

    vector<int> candidate_list;
    vector<pair<int, double>> ordered_fhps;

    clock_t start_time = clock();
    samba_first_phase_for_topk_group(source, graph, k, candidate_list, ordered_fhps);

    if (candidate_list.size() <= k)
    {
        cout << "have found the topk group " << endl;
        save_estimated_top_k(source, ordered_fhps, k);
        return ;
    }
    else
    {
        samba_second_phase_sqrt_walk_for_topk_group(source, graph, k, candidate_list);
    }
    clock_t end_time = clock();
    avg_time += (end_time-start_time)/(double) CLOCKS_PER_SEC;

    // if(config.sqrt_walk){
    // }else{
    //     samba_second_phase_rev(source, graph, k, candidate_list);
    // }

    return ;
}

void mc_fhp_ground_truth(const Graph& graph, int source, unordered_map<int, double>& map_fhps){
    static thread_local iMap<unsigned long long> rw_counter_thread_local;
    rw_counter_thread_local.initialize(graph.n);
    rw_counter_thread_local.reset_zero_values();
    //cout<<"current source: " << source << endl;
    static thread_local vector<double> fhps;
    //vector<pair<int, double>> ordered_fhps;
    static thread_local vector<bool> visited;
    for (int i = 0; i < graph.n; i++) {
        fhps.push_back(0.0);
        map_fhps.insert(make_pair(i, 0.0));
        //map_fhps.push_back();
        visited.push_back(false);
    }

    unsigned long long num_rw = graph.n * log(2 * graph.n) / config.epsilon / config.epsilon;//for ground truth, epsilon should be 0.05
    cout << "# walks: " << num_rw << endl;

    for (unsigned long long j = 0; j < num_rw; j++) {
        //a random walk from source
        //cout<<j <<endl;
        int temp_node = source;
        //int step = 0;
        vector<int> single_walk;
        single_walk.push_back(source);
        visited[source] = true;
        while (drand() > config.alpha) {
            // step += 1;
            // double incre = pow(sqrt(1 - config.alpha), step);
            if (graph.g[temp_node].size()) {
                int next = lrand() % graph.g[temp_node].size();
                temp_node = graph.g[temp_node][next];
                single_walk.push_back(temp_node);
                if (visited[temp_node] == false) {
                    if (!rw_counter_thread_local.exist(temp_node))
                        rw_counter_thread_local.insert(temp_node, 1);
                    else
                        rw_counter_thread_local[temp_node] += 1;

                    visited[temp_node] = true;
                }

            } else {
                break;
            }

        }

        for (int l = 0; l < single_walk.size(); l++) {
            int visited_node = single_walk[l];
            visited[visited_node] = false;
        }

    }

    //estimate the fhp for all nodes
    cout << "estimating fhps " << endl;
    for (int i = 0; i < graph.n; i++) {
        if (rw_counter_thread_local.exist(i) && i != source) {
            //cout<<"current i: " <<i<<endl;
            fhps[i] = rw_counter_thread_local[i] / (double) num_rw;
            map_fhps[i] = fhps[i];
        }
        else if(i == source){
            map_fhps[i] = 1.0;
        }
    }
    fhps.clear();
    visited.clear();
}

void multi_mc_fhp(const Graph& graph, const vector<int>& source, unordered_map<int, vector<pair<int ,double>>>& map_topk_fhp ){
    static thread_local unordered_map<int, double> map_fhp;
    for(int start: source){
        mc_fhp_ground_truth(graph, start, map_fhp);

        vector<pair<int ,double>> temp_top_fhp(config.k);
        partial_sort_copy(map_fhp.begin(), map_fhp.end(), temp_top_fhp.begin(), temp_top_fhp.end(),
                          [](pair<int, double> const& l, pair<int, double> const& r){return l.second > r.second;});

        map_fhp.clear();
        map_topk_fhp[start] = temp_top_fhp;
    }
}

void gen_exact_topk_fhp(const Graph& graph){
    // config.epsilon = 0.5;
    // montecarlo_setting();

    vector<int> queries;
    load_ss_query(queries);
    INFO(queries.size());
    unsigned int query_size = queries.size();
    query_size = min( query_size, config.query_size );
    INFO(query_size);

    assert(config.k < graph.n-1);
    assert(config.k > 1);
    INFO(config.k);


    split_line();
    // montecarlo_setting();

    unsigned NUM_CORES = std::thread::hardware_concurrency()-1;
    INFO(NUM_CORES);
    assert(NUM_CORES >= 2);

    int num_thread = min(query_size, NUM_CORES);
    int avg_queries_per_thread = query_size/num_thread;

    vector<vector<int>> source_for_all_core(num_thread);
    vector<unordered_map<int, vector<pair<int ,double>>>> fhp_for_all_core(num_thread);

    for(int tid=0; tid<num_thread; tid++){
        int s = tid*avg_queries_per_thread;
        int t = s+avg_queries_per_thread;

        if(tid==num_thread-1)
            t+=query_size%num_thread;

        for(;s<t;s++){
           // cout << s+1 <<". source node:" << queries[s] << endl;
            source_for_all_core[tid].push_back(queries[s]);
        }
    }

    //a thread contains more than 1 source node

    {
        Timer timer(PI_QUERY);
        INFO("mc simulation..");
        std::vector< std::future<void> > futures(num_thread);
        for(int tid=0; tid<num_thread; tid++){
            //cout<<"tid: " << tid <<endl;
            futures[tid] = std::async( std::launch::async, multi_mc_fhp, std::ref(graph), std::ref(source_for_all_core[tid]), std::ref(fhp_for_all_core[tid]) );
        }
        std::for_each(futures.begin(), futures.end(), std::mem_fn(&std::future<void>::wait));
    }

    // cout << "average iter times:" << num_iter_topk/query_size << endl;
    cout << "average generation time (s): " << Timer::used(PI_QUERY)*1.0/ (double) query_size << endl;

    stringstream ss1;
    ss1 << "real_fhp/" << config.graph_alias << "/mc_fhp_"<<config.epsilon<<"_.time";
    string outfile = ss1.str();
    ofstream time_file(outfile);
    time_file << Timer::used(PI_QUERY)*1.0/(double) query_size << endl;
    time_file.close();

    INFO("combine results...");
    INFO("storing the result for each source node...");
    /*for(int tid=0; tid<num_thread; tid++){
        for(auto &ppv: ppv_for_all_core[tid]){
            exact_topk_pprs.insert( ppv );
        }
        ppv_for_all_core[tid].clear();
    }*/

    for(int tid=0; tid<num_thread; tid++){
        int s = tid*avg_queries_per_thread;
        int t = s+avg_queries_per_thread;

        if(tid==num_thread-1)
            t+=query_size%num_thread;

        for(;s<t;s++){
            cout << s+1 <<". source node:" << queries[s] << endl;
            //source_for_all_core[tid].push_back(queries[s]);
            //storing the results for this source
            vector<pair<int, double>> temp_fhps =  fhp_for_all_core[tid][queries[s]];
            stringstream ss;
            if(config.epsilon == 0.2){
                ss << "real_fhp/" << config.graph_alias << "/mc_results/"<<to_str(queries[s])<<".txt";
            }
            else{
                ss << "real_fhp/" << config.graph_alias << "/mc_results_2/"<<to_str(queries[s])<<".txt";
            }
            string outfile = ss.str();
            ofstream topk_file(outfile);

            for(int i=0; i<temp_fhps.size(); i++){
                pair<int, double> ordered_fhp = temp_fhps[i];
                topk_file<<ordered_fhp.first << " "<< ordered_fhp.second<<endl;
            }

            topk_file.close();
        }
        fhp_for_all_core[tid].clear();
    }
    //save_exact_topk_ppr();
}

void mc_topk_fhp_ground_truth(int source, const Graph& graph){
    if(config.epsilon != 0.05){
        config.epsilon = 0.05; //the ground truth with epsilon 0.05
    }
    static thread_local iMap<unsigned long long> rw_counter_thread_local;
    rw_counter_thread_local.initialize(graph.n);
    rw_counter_thread_local.reset_zero_values();

    vector<double> fhps;
    vector<pair<int, double>> ordered_fhps;
    vector<bool> visited;
    for (int i = 0; i < graph.n; i++) {
        fhps.push_back(0.0);
        ordered_fhps.push_back(make_pair(i, 0.0));
        visited.push_back(false);
    }

    unsigned long long num_rw = graph.n * log(2 * graph.n) / config.epsilon / config.epsilon;//for ground truth, epsilon should be 0.1 or 0.2 (smaller than 0.5)
    cout << "# walks: " << num_rw << endl;

    clock_t start = clock();
    for (unsigned long long j = 0; j < num_rw; j++) {
        //a random walk from source
        int temp_node = source;
        //int step = 0;
        vector<int> single_walk;
        single_walk.push_back(source);
        visited[source] = true;
        while (random_double() > config.alpha) {
            // step += 1;
            // double incre = pow(sqrt(1 - config.alpha), step);
            if (graph.g[temp_node].size()) {
                int next = random_long(0, graph.g[temp_node].size()-1);
                temp_node = graph.g[temp_node][next];
                single_walk.push_back(temp_node);
                if (visited[temp_node] == false) {
                    if (!rw_counter_thread_local.exist(temp_node))
                        rw_counter_thread_local.insert(temp_node, 1);
                    else
                        rw_counter_thread_local[temp_node] += 1;

                    visited[temp_node] = true;
                }
                /* if(temp_node == target){
                     break;

                 }
                 */

            } else {
                break;
            }

        }

        for (int l = 0; l < single_walk.size(); l++) {
            int visited_node = single_walk[l];
            visited[visited_node] = false;
        }

    }

    //estimate the fhp for all nodes
    cout << "estimating fhps " << endl;
    for (int i = 0; i < graph.n; i++) {
        if (rw_counter_thread_local.exist(i) && i != source) {
            //cout<<"current i: " <<i<<endl;
            fhps[i] = rw_counter_thread_local[i] / (double) num_rw;
            ordered_fhps[i] = make_pair(i, fhps[i]);
        }
        else if(i == source){
            ordered_fhps[i] = make_pair(i, 1.0);
        }
    }

    clock_t end = clock();

    avg_time += (end - start)/(double) CLOCKS_PER_SEC;
    //sorting the nodes in decreasing of fhps
    cout << "sorting fhps" << endl;
    sort(ordered_fhps.begin(), ordered_fhps.end(),
         [](pair<int, double> const &l, pair<int, double> const &r) { return l.second > r.second; });


    stringstream ss;
    //if(config.epsilon == 0.2){
        ss << "real_fhp/" << config.graph_alias << "/"<<to_str(source)<<".txt";
   // }
   // else{
      //  ss << "real_fhp/" << config.graph_alias << "/mc_results_2/"<<to_str(source)<<".txt";
    //}
    string outfile = ss.str();
    ofstream topk_file(outfile);

    for(int i=0; i<config.k; i++){
        pair<int, double> ordered_fhp = ordered_fhps[i];
        topk_file<<ordered_fhp.first << " "<< ordered_fhp.second<<endl;
    }

    topk_file.close();
}

void s_mc_topk_fhp(vector<int> nodeList, const Graph& graph){
    for(int i = 0; i < nodeList.size(); i++){
        int tempSource = nodeList[i];
        mc_topk_fhp_ground_truth(tempSource, graph);
        cout <<tempSource << "done!"  << endl;
    }
}

void multi_mc_topk_fhp(const Graph& graph, int num_thread=25){
    struct timeval t_start,t_end;
    gettimeofday(&t_start, NULL);
    long start = ((long)t_start.tv_sec)*1000+(long)t_start.tv_usec/1000;

    vector<int> queries;
    load_ss_query(queries);
    INFO(queries.size());
    unsigned int query_size = queries.size();
    query_size = min( query_size, config.query_size );
    INFO(query_size);

    assert(config.k < graph.n-1);
    assert(config.k > 1);
    INFO(config.k);


    split_line();

    if(query_size < num_thread){
        num_thread = query_size;
    }
    vector<thread> threads;
    for(int i = 0; i < num_thread-1; i++){
        vector<int> s_nodes;
        for(int j = 0; j < query_size / num_thread; j++){
            s_nodes.push_back(queries[i * query_size / num_thread + j]);
        }
        threads.push_back(thread(s_mc_topk_fhp, s_nodes, graph));
    }
    vector<int> s_nodes;
    for(int j = 0; j < query_size / num_thread; j++){
        s_nodes.push_back(queries[(num_thread-1) * query_size / num_thread + j]);
    }
    s_mc_topk_fhp(s_nodes, graph);
    for (int i = 0; i < num_thread - 1; i++){
        threads[i].join();
    }
    gettimeofday(&t_end, NULL);
    long end = ((long)t_end.tv_sec)*1000+(long)t_end.tv_usec/1000;
    int cost_time = end - start;

    cout << "cost: " << cost_time / (double) 1000 << endl;
    stringstream ss;
    ss << "real_fhp/" << config.graph_alias << "/mc_fhp_"<<config.epsilon<<"_.time";
    string outfile = ss.str();
    ofstream timefile(outfile);
    timefile << cost_time / (double) 1000 /(double) query_size << endl;
    timefile.close();

}

void query(Graph& graph){

    int used_counter=0;

    if(config.algo == MC) //mc_pairwise
    { //mc
        montecarlo_setting();
        display_setting();
        // used_counter = MC_QUERY;


        vector<int> ss_queries;
        load_ss_query(ss_queries);


        vector<vector<int>> t_vec;
        load_group(t_vec, config.gsize, config.is_cluster);

        unsigned int ss_query_size = ss_queries.size();
        ss_query_size = min(ss_query_size, config.query_size);

        string method = "mc";
        stringstream ss1;
        ss1 << "estimated_fhp/" << config.graph_alias << "/pairwise/" << to_str(config.gsize) <<"/" << method<< ".txt";
        string outfile = ss1.str();
        cout<<outfile<<endl;
        ofstream est_fhp_file(outfile);

        stringstream ss;
        ss << "estimated_fhp/" << config.graph_alias << "/pairwise/"<< to_str(config.gsize) <<"/" << method<<".time";

        string outfile1 = ss.str();
        ofstream time_file(outfile1);

        avg_time = 0.0;
        for(int i=0; i<ss_query_size; i++){
            cout << i+1 <<" source: " << ss_queries[i] <<" target set size: " << t_vec[i].size() << endl;
            double cur_time = 0.0;
            double cur_fhp = mc_group_fhp(ss_queries[i], t_vec[i], graph, cur_time);
            est_fhp_file<<cur_fhp<<endl;
            time_file << cur_time <<endl;
            split_line();
        }

        est_fhp_file.close();
        cout<<"avg time: " << avg_time / (double) ss_query_size << endl;

        time_file<<"avg time: " << avg_time / (double) ss_query_size << endl;
        time_file<<"current epsilon: "<< config.epsilon<<endl;
        time_file.close();

    }
    else if (config.algo == SAMBA)//samba pairwise
    {
        samba_target_group_setting(graph.n, graph.m, config.gsize);
        display_setting();
        //used_counter = HUBPPR_QUERY;

        bwd_idx.first.initialize(graph.n);
        bwd_idx.second.initialize(graph.n);
        //hub_counter.initialize(graph.n);
        sqrt_rw_counter.initialize(graph.n);

        vector<int> ss_queries;
        load_ss_query(ss_queries);

        //int gs[4] = {20, 40, 80, 160};
        //for(int i=0; i < 4; i++){
           // int cur_gsize = gs[i];
            vector<vector<int>> t_vec;
            load_group(t_vec, config.gsize, config.is_cluster);
            unsigned int ss_query_size = ss_queries.size();
            ss_query_size = min(ss_query_size, config.query_size);

            string method = "samba";
            stringstream ss1;
            ss1 << "estimated_fhp/" << config.graph_alias << "/pairwise/"<< to_str(config.gsize) << "/" <<method<< ".txt";
            string outfile1 = ss1.str();
            cout<<outfile1<<endl;
            ofstream est_fhp_file(outfile1);

            cout <<"alpha: " << config.alpha << endl;
            cout <<"epsilon: " << config.epsilon <<endl;
            int disconnected_num = 0;
            avg_time = 0;

        stringstream ss;
        ss << "estimated_fhp/" << config.graph_alias << "/pairwise/" << to_str(config.gsize) << "/" << method<<".time";

        string outfile = ss.str();
        ofstream time_file(outfile);

            for(int i=0; i<ss_query_size; i++){
                cout << i+1 <<" source: " << ss_queries[i] <<" target set size: " << t_vec[i].size() << endl;
                double cur_time = 0.0;
                double cur_fhp = samba_query_group_sqrt_walk(ss_queries[i], t_vec[i], graph, cur_time);
                est_fhp_file << cur_fhp <<endl;
                time_file << cur_time<<endl;
                split_line();
            }

            est_fhp_file.close();


            cout<<"avg time: " << avg_time / (double) ss_query_size << endl;
            time_file<<"avg time: " << avg_time / (double) ss_query_size << endl;
            time_file<<"current epsilon: "<< config.epsilon<<endl;
            time_file.close();

    }
    else if (config.algo == FORWARD) //fora pairwise
    {
        fora_setting(graph.n, graph.m);
        display_setting();
        used_counter = FORA_QUERY;
        fwd_idx.first.nil = -1;
        fwd_idx.second.nil =-1;
        fwd_idx.first.initialize(graph.n);
        fwd_idx.second.initialize(graph.n);

        vector<int> ss_queries;
        load_ss_query(ss_queries);
        vector<vector<int>> t_vec;
        load_group(t_vec, config.gsize, config.is_cluster);
        unsigned int ss_query_size = ss_queries.size();
        ss_query_size = min(ss_query_size, config.query_size);

        int invalid_num = 0;
        avg_time = 0;
        cout <<"alpha: " << config.alpha << endl;
        cout <<"epsilon: " << config.epsilon <<endl;

        string method = "ad_fora";
        stringstream ss1;
        ss1 << "estimated_fhp/" << config.graph_alias << "/pairwise/"<< to_str(config.gsize) << "/"<< method<< ".txt";
        string outfile1 = ss1.str();
        cout<<outfile1<<endl;
        ofstream est_fhp_file(outfile1);

        stringstream ss;
        ss << "estimated_fhp/" << config.graph_alias << "/pairwise/"<< to_str(config.gsize) << "/" << method<<".time";

        string outfile = ss.str();
        ofstream time_file(outfile);

        for(int i=0; i<ss_query_size; i++){
            cout << i+1 <<" source: " << ss_queries[i] <<" target set size: " << t_vec[i].size() << endl;
            bool valid_query = true;
            double cur_time = 0.0;
            double cur_fhp = forward_push_query_group(ss_queries[i], t_vec[i], graph, valid_query, cur_time);

            if (!valid_query)
            {
                invalid_num++;
            }

            est_fhp_file<< cur_fhp <<endl;
            time_file<<cur_time<<endl;

            split_line();
        }
        est_fhp_file.close();

        int valid_num = ss_query_size-invalid_num;
        cout<<"valid num: " << valid_num << ", avg time: " << avg_time / (double) (valid_num) << endl;
        time_file<<"avg time: " << avg_time / (double) (ss_query_size-invalid_num)<< endl;
        time_file<<"current epsilon: "<< config.epsilon<<endl;
        time_file.close();
    }
    else if (config.algo == POWER)//power pairwise
    {
        vector<int> ss_queries;
        load_ss_query(ss_queries);
        vector<vector<int>> t_vec;
        load_group(t_vec, config.gsize, config.is_cluster);
        unsigned int ss_query_size = ss_queries.size();
        ss_query_size = min(ss_query_size, config.query_size);

        int invalid_num = 0;
        avg_time = 0;
        cout <<"alpha: " << config.alpha << endl;
        cout <<"epsilon: " << config.epsilon <<endl;

        string method = "power";
        stringstream ss1;
        ss1 << "estimated_fhp/" << config.graph_alias << "/pairwise/"<< to_str(config.gsize) << "/"<< method<< ".txt";
        string outfile1 = ss1.str();
        cout<<outfile1<<endl;
        ofstream est_fhp_file(outfile1);

        stringstream ss;
        ss << "estimated_fhp/" << config.graph_alias << "/pairwise/"<< to_str(config.gsize) << "/" << method<<".time";

        string outfile = ss.str();
        ofstream time_file(outfile);

        for(int i=0; i<ss_query_size; i++){
            cout << i+1 <<" source: " << ss_queries[i] <<" target set size: " << t_vec[i].size() << endl;
            //bool valid_query = true;
            double cur_time = 0.0;
            double cur_fhp =0.0;
            fhp_power_iteration(graph, ss_queries[i], t_vec[i], cur_fhp, cur_time);

            est_fhp_file<< cur_fhp <<endl;
            time_file<<cur_time<<endl;

            split_line();
        }
        est_fhp_file.close();

        //int valid_num = ss_query_size-invalid_num;
        cout<< "avg time: " << avg_time / (double) (ss_query_size) << endl;
        time_file<<"avg time: " << avg_time / (double) (ss_query_size)<< endl;
        time_file<<"current epsilon: "<< config.epsilon<<endl;
        time_file.close();
    }
}

void topk_query(Graph& graph){

    int used_counter=0;

    if(config.algo == FAST_PRUNE){
        INFO(config.algo);
        vector<int> queries;
        //load_target(queries);
        load_ss_query(queries);
        avg_time = 0;

        unsigned int query_size = queries.size();
        query_size = min( query_size, config.query_size );
        INFO(query_size);

        bippr_setting(graph.n, graph.m);
        display_setting();
        //used_counter = BIPPR_QUERY;

        bwd_idx.first.initialize(graph.n);
        bwd_idx.second.initialize(graph.n);

        rw_counter.initialize(graph.n);
        sqrt_rw_counter.initialize(graph.n);


        string method ="fastPrune";
        if(!config.sqrt_walk){
            method = "fastPrune1";
        }
        stringstream ss;
        ss << "estimated_fhp/" << config .graph_alias << "/topk/"<<method <<"/" <<to_str(config.epsilon) << "_results.txt";
        string outfile = ss.str();
        ofstream topk_file(outfile);
        topk_file <<"k time prec" <<endl;

        //int ks[5] = {100, 200, 300, 400, 500};
        //for(int i=0; i<5; i++){
            //int cur_k = ks[i];
        int cur_k =config.k;
            avg_time = 0.0;
            double avg_prec = 0.0;
            for(int i=0; i<query_size; i++){
                cout << i+1 <<". source nodes: " << queries[i] << endl;
                clock_t bfs_time = clock();
                double larger_k = do_bfs(graph, queries[i], cur_k, method);
                clock_t bfs_end_time = clock();
                avg_time += (bfs_end_time-bfs_time)/(double) CLOCKS_PER_SEC;
                if(larger_k){
                    //samba_topk(queries[i],graph, cur_k);
                    fast_prune(queries[i],graph, cur_k);
                }

                //cout<<"results.size: " << results.size()<<endl;
                double prec = compute_precision(queries[i], method, cur_k, config.epsilon);
                cout<<"precision: " << prec <<endl;
                avg_prec += prec;
                split_line();
            }


            cout<<"avg time: " << avg_time / (double) query_size << endl;
            cout<<"avg precision: " << avg_prec / (double) query_size << endl;

            topk_file<<cur_k <<" " <<avg_time / (double) query_size <<" " << avg_prec / (double) query_size << endl;


        //}

        topk_file.close();

    }
    else if(config.algo == FAST_PRUNE_GROUP){
        INFO(config.algo);
        avg_time = 0;

        vector<int> queries;
        load_ss_query(queries);

        load_group(groups_list, config.gsize, config.is_cluster);
        build_group_inv_list(groups_list, graph.n);

        unsigned int query_size = queries.size();
        query_size = min( query_size, config.query_size);
        INFO(query_size);

        bippr_setting(graph.n, graph.m);
        display_setting();
        //used_counter = BIPPR_QUERY;

        bwd_idx.first.initialize(graph.n);
        bwd_idx.second.initialize(graph.n);

        rw_counter.initialize(graph.n);
        sqrt_rw_counter.initialize(graph.n);

        string outfile = build_topk_statistics_file_path();
        ofstream topk_file(outfile);
        topk_file <<"k time prec" <<endl;

        int ks[5] = {20, 40, 60, 80, 100};
        for(int i=0; i<5; i++)
        {
            int cur_k = ks[i];
            //int cur_k =config.k;
            config.k = cur_k;
            cout << "k value: " << cur_k << endl;

            avg_time = 0.0;
            double avg_prec = 0.0;
            int valid_query = 0;
            for(int i=0; i<query_size; i++){
                cout << i+1 <<". source nodes: " << queries[i] << endl;
                clock_t bfs_time = clock();
                double larger_k = do_group_bfs(graph, queries[i], cur_k);
                clock_t bfs_end_time = clock();
                //avg_time += (bfs_end_time-bfs_time)/(double) CLOCKS_PER_SEC;
                if(larger_k){
                    cout << "run fast prune group" << endl;
                    fast_prune_group(queries[i], graph, cur_k);
                    valid_query++;
                }

                //cout<<"results.size: " << results.size()<<endl;
                // double prec = compute_precision(queries[i], method, cur_k, config.epsilon);
                // cout<<"precision: " << prec <<endl;
                //avg_prec += prec;
                split_line();
            }


            cout<<"avg time: " << avg_time / (double) valid_query <<", valid queries: " << valid_query << endl;
            cout<<"avg precision: " << avg_prec / (double) valid_query << endl;

            topk_file<<cur_k <<" " <<avg_time / (double) valid_query <<" " << avg_prec / (double) valid_query << endl;


        }

        topk_file.close();

    }

    else if(config.algo == MC_TOPK){ //mc with the early termination condition
        INFO(config.algo);
        vector<int> queries;
        //load_target(queries);
        load_ss_query(queries);
        avg_time = 0;

        unsigned int query_size = queries.size();
        query_size = min( query_size, config.query_size );
        INFO(query_size);

        display_setting();

        sqrt_rw_counter.initialize(graph.n);

        // vector<pair<int, double> > results; //the top_k results;
        string method ="mc_topk";
        stringstream ss;
        ss << "estimated_fhp/" << config.graph_alias << "/topk/"<< method << "/"<<to_str(config.epsilon) <<"_results.txt";
        string outfile = ss.str();
        ofstream topk_file(outfile);
        topk_file <<"k time prec" <<endl;

        //int ks[5] = {100, 200, 300, 400, 500};
        //for(int i=0; i<5; i++){
           // int cur_k = ks[i];
            int cur_k = config.k;
            avg_time = 0.0;
            double avg_prec = 0.0;
            for(int i=0; i<query_size; i++){
                cout << i+1 <<". source nodes: " << queries[i] << endl;
                clock_t bfs_time = clock();
                bool larger_k = do_bfs(graph, queries[i], cur_k, method);
                clock_t bfs_end_time = clock();
                avg_time += (bfs_end_time-bfs_time)/(double) CLOCKS_PER_SEC;
                if(larger_k) {
                    mc_topk_early(queries[i], graph, cur_k);
                }
                //cout<<"results.size: " << results.size()<<endl;
                double prec = compute_precision(queries[i], method, cur_k, config.epsilon);
                cout<<"precision: " << prec <<endl;
                avg_prec += prec;
                split_line();
            }


            cout<<"avg time: " << avg_time / (double) query_size << endl;
            cout<<"avg precision: " << avg_prec / (double) query_size << endl;

            topk_file<<cur_k <<" " <<avg_time / (double) query_size <<" " << avg_prec / (double) query_size << endl;


        //}

        topk_file.close();
    }
    else if(config.algo == MC){ //mc for top-k query
        INFO(config.algo);
        vector<int> queries;
        //load_target(queries);
        load_ss_query(queries);
        avg_time = 0;

        unsigned int query_size = queries.size();
        query_size = min( query_size, config.query_size );
        INFO(query_size);
        montecarlo_setting();
        display_setting();
        //used_counter = MC_QUERY;

        rw_counter.initialize(graph.n);

        string method ="mc";
        stringstream ss;
        ss << "estimated_fhp/" << config .graph_alias << "/topk/"<< method << "/"<<to_str(config.epsilon) <<"_results.txt";
        string outfile = ss.str();
        ofstream topk_file(outfile);
        topk_file <<"k time prec" <<endl;

        //int ks[5] = {100, 200, 300, 400, 500};
        //for(int i=0; i<5; i++){
        // int cur_k = ks[i];
        int cur_k = config.k;
        avg_time = 0.0;
        double avg_prec = 0.0;
        for(int i=0; i<query_size; i++){
            cout << i+1 <<". source nodes: " << queries[i] << endl;
            clock_t bfs_time = clock();
            bool larger_k = do_bfs(graph, queries[i], cur_k, method);
            clock_t bfs_end_time = clock();
            avg_time += (bfs_end_time-bfs_time)/(double) CLOCKS_PER_SEC;
            if(larger_k) {
                mc_topk_fhp(queries[i], graph);
            }
            //cout<<"results.size: " << results.size()<<endl;
            // double prec = compute_precision(queries[i], method, cur_k, config.epsilon);
            // cout<<"precision: " << prec <<endl;
            // avg_prec += prec;
            split_line();
        }


        cout<<"avg time: " << avg_time / (double) query_size << endl;
        cout<<"avg precision: " << avg_prec / (double) query_size << endl;

        topk_file<<cur_k <<" " <<avg_time / (double) query_size <<" " << avg_prec / (double) query_size << endl;


        //}

        topk_file.close();

        /*
        //cout <<"alpha: " << config.alpha<< endl;
        //unsigned long source = 364102;
        //vector<double> dhts;
        for(int i=0; i<query_size; i++){
            cout << i+1 <<". source node:" << queries[i] << endl;
            mc_topk_fhp(queries[i], graph);

            split_line();
        }

        stringstream ss;
        ss << "estimated_fhp/" << config.graph_alias << "/topk/mc/exp.txt";
        string outfile = ss.str();
        ofstream time_file(outfile);
        cout<<"avg time: " << avg_time / (double) query_size << endl;
        time_file<<"avg time: " << avg_time / (double) query_size << endl;
        time_file<<"current epsilon: "<< config.epsilon<<endl;
        time_file.close();
*/
    }
    else if(config.algo == MC_GROUP){ //mc for top-k group query
        INFO(config.algo);
        avg_time = 0;

        vector<int> queries;
        load_ss_query(queries);

        load_group(groups_list, config.gsize, config.is_cluster);
        build_group_inv_list(groups_list, graph.n);

        unsigned int query_size = queries.size();
        query_size = min( query_size, config.query_size );
        INFO(query_size);
        montecarlo_setting();
        display_setting();
        //used_counter = MC_QUERY;

        rw_counter.initialize(groups_list.size());

        string outfile = build_topk_statistics_file_path();
        ofstream topk_file(outfile);
        topk_file <<"k time prec" <<endl;

        int ks[5] = {20, 40, 60, 80, 100};
        for(int i=0; i<5; i++)
        {
            int cur_k = ks[i];
            //int cur_k = config.k;
            config.k = cur_k;
            cout << "k value: " << cur_k << endl;

            avg_time = 0.0;
            double avg_prec = 0.0;
            for(int i=0; i<query_size; i++){
                cout << i+1 <<". source nodes: " << queries[i] << endl;
                clock_t bfs_time = clock();
                double larger_k = do_group_bfs(graph, queries[i], cur_k);
                clock_t bfs_end_time = clock();
                avg_time += (bfs_end_time-bfs_time)/(double) CLOCKS_PER_SEC;
                if(larger_k) {
                    mc_topk_group(queries[i], graph);
                }
                //cout<<"results.size: " << results.size()<<endl;
                // double prec = compute_precision(queries[i], method, cur_k, config.epsilon);
                // cout<<"precision: " << prec <<endl;
                // avg_prec += prec;
                split_line();
            }


            cout<<"avg time: " << avg_time / (double) query_size << endl;
            cout<<"avg precision: " << avg_prec / (double) query_size << endl;

            topk_file<<cur_k <<" " <<avg_time / (double) query_size <<" " << avg_prec / (double) query_size << endl;

        }
        topk_file.close();
    }
    else if(config.algo == MC_TOPK_GROUP){ //mc with the early termination condition
        INFO(config.algo);
        avg_time = 0;

        vector<int> queries;
        load_ss_query(queries);

        load_group(groups_list, config.gsize, config.is_cluster);
        build_group_inv_list(groups_list, graph.n);

        unsigned int query_size = queries.size();
        query_size = min( query_size, config.query_size );
        INFO(query_size);
        montecarlo_setting();
        display_setting();
        //used_counter = MC_QUERY;

        rw_counter.initialize(groups_list.size());

        string outfile = build_topk_statistics_file_path();
        ofstream topk_file(outfile);
        topk_file <<"k time prec" <<endl;

        int ks[5] = {20, 40, 60, 80, 100};
        for(int i=0; i<5; i++)
        {
            int cur_k = ks[i];
            config.k = cur_k;
            cout << "k value: " << cur_k << endl;

            //int cur_k = config.k;
            avg_time = 0.0;

            double avg_prec = 0.0;
            int valid_query =0;
            for(int i=0; i<query_size; i++){
                cout << i+1 <<". source nodes: " << queries[i] << endl;
                clock_t bfs_time = clock();
                double larger_k = do_group_bfs(graph, queries[i], cur_k);
                clock_t bfs_end_time = clock();
                //avg_time += (bfs_end_time-bfs_time)/(double) CLOCKS_PER_SEC;
                if(larger_k)
                {
                    mc_topk_group_early(queries[i], graph, cur_k);
                    valid_query++;
                }
                //cout<<"results.size: " << results.size()<<endl;
                // double prec = compute_precision(queries[i], method, cur_k, config.epsilon);
                // cout<<"precision: " << prec <<endl;
                // avg_prec += prec;
                split_line();
            }


            cout<<"avg time: " << avg_time / (double) valid_query <<", valid queries: " << valid_query<< endl;
            cout<<"avg precision: " << avg_prec / (double) valid_query << endl;

            topk_file<<cur_k <<" " <<avg_time / (double) valid_query <<" " << avg_prec / (double) valid_query << endl;
        }
        topk_file.close();
    }
}

#endif //FHP_QUERY_H
