
#ifndef FHP_GROUP_H
#define FHP_GROUP_H

#include "config.h"

extern double avg_time;
extern vector<vector<int>> groups_inverted_list;
string build_topk_result_file_path(int k, int source);

bool do_bfs(const Graph& graph, int start, int k, string method)
{
    std::vector<bool> visited_vec(graph.n);
    std::vector<int> queue;
    int visited_num = 0;

    queue.push_back(start);
    visited_vec[start] = true;
    int curr = 0;
    int degree = graph.g[start].size();
    int level_start = start;
    int level = 0;

    while(curr <= visited_num)
    {
        int node = queue[curr];

        if (level_start == node)
        {
            level++;
            level_start = -1;
        }

        for(auto &nbr: graph.g[node])
        {
            if (!visited_vec[nbr])
            {
                queue.push_back(nbr);
                visited_num++;
                // INFO(visited_num, nbr);
                degree += graph.g[nbr].size();
                if (level_start < 0)
                {
                    level_start = nbr;
                }
                visited_vec[nbr] = true;
            }

        }

        if (visited_num >= k)
        {
            // INFO("bfs number: ", visited_num);
            // INFO("average degree: ", degree*1.0/visited_num);
            // INFO("level: ", level);

            return true;
        }
        curr++;
    }

    //INFO("bfs number: ", visited_num);
    stringstream ss;
    ss << "estimated_fhp/" << config.graph_alias << "/topk/"<< method <<"/" <<k<<"/"<<to_str(start)<<".txt";
    string outfile = ss.str();
    ofstream topk_file(outfile);

    for(int i=0; i<graph.n; i++){
        if(visited_vec[i]){
            topk_file<<i << " "<< 1.0 <<endl;
        }

    }
    topk_file.close();

    return false;
}

bool do_group_bfs(const Graph& graph, int start, int k)
{
    std::vector<bool> visited_vec(graph.n);
    std::vector<int> queue;
    int visited_num = 0;

    queue.push_back(start);
    visited_vec[start] = true;
    int curr = 0;
    int degree = graph.g[start].size();
    int level_start = start;
    int level = 0;

    unordered_set<int> group_index_set;
    while(curr <= visited_num)
    {
        int node = queue[curr];


        for(auto &nbr: graph.g[node])
        {
            if (!visited_vec[nbr])
            {
                queue.push_back(nbr);
                visited_num++;
                // INFO(visited_num, nbr);
                //degree += graph.g[nbr].size();

                visited_vec[nbr] = true;
                for (auto& group_index: groups_inverted_list[nbr])
                {
                    if (group_index_set.find(group_index) == group_index_set.end())
                    {
                        group_index_set.insert(group_index);
                    }
                }
            }

        }

        if (group_index_set.size() >= k)
        {
            // INFO("bfs number: ", visited_num);
            // INFO("average degree: ", degree*1.0/visited_num);
            // INFO("level: ", level);

            return true;
        }
        curr++;
    }

    INFO("bfs number: ", visited_num);

    string outfile = build_topk_result_file_path(k, start);
    ofstream topk_file(outfile);

    for(auto index: group_index_set){
            topk_file<<index << " "<< 1.0 <<endl;
    }
    topk_file.close();

    return false;
}

inline void generate_cluster(int start, const Graph& graph, int gsize, unordered_set<int>& t_set)
{
    int max_iter = 10000;
    t_set.insert(start);
    int iter = 0;
    while(t_set.size() < gsize && iter < max_iter)
    {

        int dst = random_walk(start, graph);
        if (t_set.find(dst) == t_set.end())
        {
            t_set.insert(dst);
        }

        iter++;
    }
}

inline void make_shuffle(vector<int>& shuf, int n)
{
    // Initialize seed randomly
    shuf.resize(n);
    for (int i = 0; i < n; i++)
    {
        shuf[i] = i;
    }

    for (int i=0; i<n ;i++)
    {
        // Random for remaining positions.
        int r = i + (rand() % (n -i));

        swap(shuf[i], shuf[r]);
    }


}

unordered_set<int> aggregate_set;
inline void generate_cluster_nbr(int start, const Graph& graph, int gsize, unordered_set<int> &t_set)
{
    int curr = start;
    int iter = 0;

    int prev_size = 0;
    while (gsize > t_set.size())
    {
        while(graph.gr[curr].size() == 0 || aggregate_set.find(curr) != aggregate_set.end())
        {
            curr = rand()%graph.n;
        }

        t_set.insert(curr);
        aggregate_set.insert(curr);
        if (t_set.size() >= gsize)
        {
            return ;
        }


        vector<int> shuffle_vec;
        auto &nbrs = graph.gr[curr];
        int nbr_size = nbrs.size();
        make_shuffle(shuffle_vec, nbr_size);

        for (int i = 0; i < nbr_size; i++)
        {
            int node = nbrs[shuffle_vec[i]];
            if (aggregate_set.find(node) != aggregate_set.end())
            {
                continue;
            }

            t_set.insert(node);
            aggregate_set.insert(node);

            if (t_set.size() >= gsize)
            {
                return ;
            }
        }
        curr = nbrs[rand()%nbr_size];
        if (iter%10 == 0)
        {
            if (prev_size == t_set.size())
            {
                curr = rand()%graph.n;
            }
            else
            {
                prev_size = t_set.size();
            }
        }
    }
}

inline void generate_group_cluster(int n, const Graph& graph, int gsize){
    //string filename = config.graph_location + "ssquery.txt";
    string filename = config.graph_location+"/"+ config.graph_alias+ "_cluster_" + to_str(gsize) +".group";
    ofstream queryfile(filename);

    aggregate_set.clear();
    for(int i=0; i<config.group_num; i++){
    //for(int i=0; i<1; i++){

        int v = rand()%n;
        while (graph.g[v].size() == 0)
        {
            v = rand()%n;
        }

        unordered_set<int> t_set;
        generate_cluster_nbr(v, graph, gsize, t_set);
        if(t_set.size() != gsize)
        {
            i--;
            continue;
        }

        int k = 0;
        for (int node: t_set)
        {
            queryfile<<node;
            k++;
            if (k < t_set.size())
            {
                queryfile << " ";
            }
            else
            {
                queryfile << endl;
            }
        }
    }
}



inline void generate_group_random(int n, int gsize)
{
    string filename = config.graph_location+"/"+ config.graph_alias+ "_random_" + to_str(gsize) +".group";
    ofstream queryfile(filename);

    cout << "generate a random group" << endl;
    for(int i=0; i<config.group_num; i++)
    {
    //for(int i=0; i<1; i++){
        unordered_set<int> t_set;

        while(t_set.size() < gsize)
        {
            int v = rand()%n;
            if (t_set.find(v) == t_set.end())
            {
                t_set.insert(v);
            }
        }

        i += gsize;

        int k = 0;
        for (int node: t_set)
        {
            queryfile<<node;
            k++;
            if (k < t_set.size())
            {
                queryfile << " ";
            }
            else
            {
                queryfile << endl;
            }
        }

    }
}

inline void generate_group(int n, const Graph& graph, int gsize, bool is_cluster)
{
    if(is_cluster)
    {
        generate_group_cluster(n, graph, gsize);
    }
    else
    {
        generate_group_random(n, gsize);
    }
}


inline void samba_target_group_setting(int n, long long m, int group_size)
{
    config.rmax = config.epsilon*sqrt(m*1.0*config.delta*group_size/3.0/log(2.0/config.pfail)/(double) n);
    config.rmax *= config.rmax_scale;
    double factor = config.rmax/config.delta;
    factor = (factor> 1)?factor:1;

    config.omega = factor*3.0*log(2.0/config.pfail)/config.epsilon/config.epsilon;
    INFO(factor);
}

inline void balance_rmax(double rmax, long long push_op, bool& time_stop)
{
    double factor = rmax/config.delta;
     factor = (factor> 1)?factor:1;

    long long omega = factor*3.0*log(2.0/config.pfail)/config.epsilon/config.epsilon;
    long long walk_op = omega/config.alpha;

    cout << "require walk operation: " << walk_op << ", push_op: " << push_op <<endl;
    if (walk_op < push_op)
    {
        time_stop = true;
        config.rmax = rmax;
        config.omega = omega;
    }
    else
    {
        time_stop = false;
    }

    return ;

}

/*inline void fora_setting(int n, long long m){
    double delta = 1/(double) n;
    double pfail = 1/ (double) n;
    config.rmax = config.epsilon*sqrt(delta/3/m/log(2/pfail));
    config.rmax *= config.rmax_scale;
    // config.rmax *= config.multithread_param;
    config.omega = (2+config.epsilon)*log(2/pfail)/delta/config.epsilon/config.epsilon;
}*/


void load_group(vector<vector<int>> &queries, int gsize, bool is_cluster)
{
    string filename;
    if (is_cluster)
    {
        filename = config.graph_location+"/"+ config.graph_alias+ "_cluster_"+to_str(gsize)+".group";
    }
    else
    {
        filename = config.graph_location+"/"+ config.graph_alias+ "_random_"+to_str(gsize)+".group";
    }

    if(!file_exists_test(filename))
    {
        cerr<<"query file does not exist, please generate group query files first: " << filename <<endl;
        exit(0);
    }

    ifstream queryfile(filename);

    string s;
    while(getline(queryfile, s))
    {
        vector<int> group;

        istringstream iss(s);
        int node = 0;
        while(iss >> node)
        {
            group.push_back(node);
        }

        if (group.size() != gsize)
        {
            cerr << "data in query file is wrong." << endl;
            exit(0);
        }

        queries.push_back(group);
    }

    cout << "group number: " << queries.size() << endl;

    return ;
}

double mc_group_fhp(int source, vector<int>& t_vec, const Graph& graph, double& cur_time){
    //for pairwise query
    //rw_counter.reset_zero_values();
    //rw stops as long as hitting target

    double fhp;
    unordered_set<int> t_set(t_vec.begin(), t_vec.end());
    clock_t start = clock();
    if(t_set.find(source) != t_set.end()){
        fhp = 1.0;
    }else {
        unsigned long rw_counter = 0;
        unsigned long num_rw = 3 * graph.n * log(2 * graph.n) / config.epsilon / config.epsilon;
        cout << "# walks: " << num_rw << endl;

        for (unsigned long i = 0; i < num_rw; i++) {
            int temp_node = source;
            while (drand() > config.alpha) {
                if (graph.g[temp_node].size()) {
                    int next = lrand() % graph.g[temp_node].size();
                    temp_node = graph.g[temp_node][next];

                    if (t_set.find(temp_node) != t_set.end()) {
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
    clock_t end = clock();
    cout<<"estimated fhp by mc: " << fhp <<endl;

    avg_time += (end - start)/(double) CLOCKS_PER_SEC;
    cur_time = (end - start)/(double) CLOCKS_PER_SEC;
    return fhp;
}

void compute_fhp_with_fwdidx(const Graph& graph, double check_rsum, unordered_set<int> &t_set){

    int node_id;
    double reserve;

    // INFO("rsum is:", check_rsum);
    if(check_rsum == 0.0)
        return;

    unsigned long long num_random_walk = config.omega*check_rsum;
    // INFO(num_random_walk);
    //num_total_rw += num_random_walk;

    //rand walk online

    double fhp = 0.0;

    for (int target: t_set)
    {
        if (fwd_idx.second.exist(target))
        {
            fhp += fwd_idx.second[target];
            fwd_idx.second[target] = 0.0;
        }
    }

    std::cout << "first stage: " << fhp <<std::endl;

    unsigned long num_visited = 0;

    for(long i=0; i < fwd_idx.second.occur.m_num; i++)
    {
        int source = fwd_idx.second.occur[i];

        double residual = fwd_idx.second[source];

        if (residual == 0.0)
        {
            continue;
        }

        unsigned long num_s_rw = ceil(residual/check_rsum*num_random_walk);
        double a_s = residual/check_rsum*num_random_walk/num_s_rw;

        double ppr_incre = a_s*check_rsum/num_random_walk;

        num_total_rw += num_s_rw;

        for (unsigned long i = 0; i < num_s_rw; i++)
        {
            int temp_node = source;
            while (drand() > config.alpha)
            {
                if (graph.g[temp_node].size())
                {
                    int next = lrand() % graph.g[temp_node].size();
                    temp_node = graph.g[temp_node][next];

                    if (t_set.find(temp_node) != t_set.end())
                    {
                        //num_visited += 1;
                        fhp += ppr_incre;
                        break;
                    }

                }
                else
                {
                    break;
                }
            }
        }
    }

    //fhp += (double)num_visited/(double)num_total_rw;
    cout << "final fhp: " << fhp << endl;
}

void forward_local_update_linear_group(int s, std::unordered_set<int> &t_set, const Graph &graph, double& rsum)
{
    fwd_idx.first.clean();
    fwd_idx.second.clean();

    static vector<bool> idx(graph.n);
    std::fill(idx.begin(), idx.end(), false);

    double myeps = config.rmax;//config.rmax;

    vector<int> q;  //nodes that can still propagate forward
    q.reserve(graph.n);
    q.push_back(-1);
    unsigned long left = 1;
    q.push_back(s);

    double init_residual = 1.0;
    fwd_idx.second.insert(s, init_residual);

    idx[s] = true;

    while (left < (int) q.size()) {
        int v = q[left];
        idx[v] = false;
        left++;
        double v_residue = fwd_idx.second[v];
        fwd_idx.second[v] = 0;

        int out_neighbor = graph.g[v].size();
        if(out_neighbor == 0){
            rsum -= v_residue;
            continue;
        }


        rsum -= v_residue*config.alpha;

        double avg_push_residual = ((1.0 - config.alpha) * v_residue) / out_neighbor;
        for (int next : graph.g[v]) {
            // total_push++;
            if( !fwd_idx.second.exist(next) )
                fwd_idx.second.insert( next,  avg_push_residual);
            else
                fwd_idx.second[next] += avg_push_residual;

            //if a node's' current residual is small, but next time it got a laerge residual, it can still be added into forward list
            //so this is correct

            if (t_set.find(next) != t_set.end())
            {
                rsum -= avg_push_residual;
                continue;
            }

            if ( fwd_idx.second[next]/graph.g[next].size() >= myeps && idx[next] != true) {
                idx[next] = true;//(int) q.size();
                q.push_back(next);
            }
        }
    }
}

static void backward_push_group(int s, vector<int>& t_vec, const Graph &graph, double myeps, double init_residual = 1)  {

    bwd_idx.first.clean();
    bwd_idx.second.clean();

    static unordered_map<int, bool> idx;
    idx.clear();

    unordered_set<int> t_set(t_vec.begin(), t_vec.end());

    vector<int> q;
    q.reserve(graph.n);
    q.push_back(-1);
    unsigned long left = 1;

    //double myeps = 0.0000000001; //10^{-10}

    for (int target: t_vec)
    {
        bwd_idx.second.insert(target, init_residual);
        q.push_back(target);
        idx[target] = true;
    }

    double remove_residual = 0.0;
    long long push_operation = 0;
    if(1) {
        //t is not an out-neighbor of s
        while (left < q.size()) {
            int v = q[left];
            idx[v] = false;
            left++;
            if (bwd_idx.second[v] < myeps)
                continue;

            double residual = (1.0 - config.alpha) * bwd_idx.second[v];
            bwd_idx.second[v] = 0;
            if (graph.gr[v].size() > 0) {
                for (int next : graph.gr[v]) {

                    double incr = residual/graph.g[next].size();

                    if (t_set.find(next) != t_set.end())
                    {
                        remove_residual += incr;
                        continue;
                    }

                    push_operation++;

                    if (bwd_idx.second.notexist(next))
                        bwd_idx.second.insert(next, incr);
                    else
                        bwd_idx.second[next] += incr;

                    if (next != s && bwd_idx.second[next] > myeps && idx[next] != true) {
                        // put next into q if next is not in q
                            idx[next] = true;//(int) q.size();
                            q.push_back(next);
                    }

                }
            }
        }
    }
    else{
       cout<<"t is an out-neighbor of s such that no backward search is performed" <<endl;
    }

    cout << "remove_residual: " << remove_residual << endl;
    cout << "push_operation: " << push_operation <<endl;

}


static void backward_push_group_balance(int s, vector<int>& t_vec, const Graph &graph, double R_max, double init_residual = 1)  {
    bwd_idx.first.clean();
    bwd_idx.second.clean();

    static unordered_map<int, bool> idx;
    idx.clear();

    unordered_set<int> t_set(t_vec.begin(), t_vec.end());

    vector<int> q;
    q.reserve(graph.n);
    q.push_back(-1);
    unsigned long left = 1;

    double myeps =1.0/t_vec.size(); //10^{-10}

    for (int target: t_vec)
    {
        bwd_idx.second.insert(target, init_residual);
        q.push_back(target);
        idx[target] = true;
    }

    bwd_idx.second.insert(s, 0);
    bwd_idx.first.insert(s, 0);//s always have reserve by converting its residues

    double remove_residual = 0.0;
    long long push_operation = 0;
    int iter = 0;
    //double source_residual = 0.0;
    while(R_max <= myeps) {
        cout << "iteration " << iter << ", eps: " << myeps <<endl;
        //t is not an out-neighbor of s
        while (left < q.size()) {
            int v = q[left];
            idx[v] = false;
            left++;
            if(v == s){//
                bwd_idx.first[s] += bwd_idx.second[v];
            }
            if (bwd_idx.second[v] < myeps)
                continue;

            double residual = (1.0 - config.alpha) * bwd_idx.second[v];
            bwd_idx.second[v] = 0;
            if (graph.gr[v].size() > 0) {
                for (int next : graph.gr[v]) {

                    double incr = residual/graph.g[next].size();

                    if (t_set.find(next) != t_set.end())
                    {
                        remove_residual += incr;
                        continue;
                    }

                    push_operation++;

                    if (bwd_idx.second.notexist(next))
                        bwd_idx.second.insert(next, incr);
                    else
                        bwd_idx.second[next] += incr;

                    if ( bwd_idx.second[next] > myeps ) {
                        /*
                        if (next == s || idx[next] == true)
                        {
                            continue;
                        }*/
                        if(idx[next] == true){
                            continue;
                        }
                        // put next into q if next is not in q
                            idx[next] = true;//(int) q.size();
                            q.push_back(next);
                    }
                }
            }
        }

        // 防止source被push
        //source_residual += bwd_idx.second[s];
        //bwd_idx.second[s] = 0.0;

        if (myeps <= R_max)
        {
            break;
        }

        bool time_stop = false;
        balance_rmax(myeps, push_operation, time_stop);
        if (time_stop)
        {

            break;
        }

        myeps = myeps/2.0;
        myeps = (R_max > myeps)?R_max:myeps;
        double percent = (double)(bwd_idx.second.occur.m_num)/graph.n;
        cout << "occur percent: " << percent << endl;
        if (percent < 0.1)
        {
            for (int j=0; j<bwd_idx.second.occur.m_num; j++)
            {
                int node = bwd_idx.second.occur[j];
                if (bwd_idx.second[node] > myeps)
                {
                    q.push_back(node);
                    idx[node] = true;
                }
            }
        }
        else
        {
            for (int j = 0; j < graph.n; j++)
            {
                if (bwd_idx.second.exist(j) && bwd_idx.second[j] > myeps)
                {
                    q.push_back(j);
                    idx[j] = true;
                }
            }
        }
    }

    //bwd_idx.second[s] = source_residual;

    cout << "remove_residual: " << remove_residual << endl;
    cout << "push_operation: " << push_operation <<endl;

}

static void reverse_local_update_linear_visiting_prob_group(int s, vector<int>& t_vec, const Graph &graph, double eps, double init_residual = 1)  {

    backward_push_group_balance(s, t_vec, graph, eps);
}

double forward_push_query_group(int source, vector<int> t_vec, const Graph& graph, bool& valid_query, double& cur_time) {
    //Timer timer(BIPPR_QUERY);
    clock_t start = clock();
    double fhp = 0.0;

    unordered_set<int> t_set(t_vec.begin(), t_vec.end());

    if (t_set.find(source) != t_set.end()) {
        fhp = 1.0;
    }
    else
    {
        //double new_rmax = 0.001; //10^{-6}
        INFO(config.rmax);
        //cout<< "rmax used: " << new_rmax <<endl;
        //first run the backward propagation from the target
        double rsum = 1.0;
        forward_local_update_linear_group(source, t_set, graph, rsum);

        //simulate the random walks from the source
        //for the j-th step of the i-th walk, add R(v,t) to the final value of h(s,t)

        //Timer tm(RONDOM_WALK);
        /*
        unsigned long long num_rw;
        int outdeg_source = graph.g[source].size();
        cout<<"out degree of source: " << outdeg_source<<endl;
        for(int i=0; i < outdeg_source; i++){
            int node_id = graph.g[source][i];
            if(bwd_idx.second.exist(node_id)){
                cout<<"residue of " << node_id << " : " << bwd_idx.second[node_id]<<endl;
            }
            else{
                cout<<"node " << node_id<<endl;
            }
        }
        double stop_prob = 1 - sqrt(1-config.alpha);
        cout<<"stop probability: " << stop_prob <<endl;

        if(find(graph.g[source].begin(), graph.g[source].end(), target) == graph.g[source].end()){
            num_rw = config.rmax_scale * 3 * new_rmax *graph.n * log(2*graph.n) / config.epsilon / config.epsilon / stop_prob ;
        }else{
            num_rw = config.rmax_scale * log(2*graph.n) * 3 * outdeg_source * log(2*graph.n) / config.epsilon / config.epsilon/stop_prob/(1-config.alpha);
        }
        cout << "# of walks: " << num_rw << endl;
        */
        INFO(config.omega, config.omega*rsum, rsum);
        if(rsum == 0.0)
        {
            valid_query = false;
            clock_t end = clock();
            avg_time += (end - start)/(double) CLOCKS_PER_SEC;
            cur_time = (end - start)/(double) CLOCKS_PER_SEC;

            return 0.0;
        }

        compute_fhp_with_fwdidx(graph, rsum, t_set);

        clock_t end = clock();
        avg_time += (end - start)/(double) CLOCKS_PER_SEC;
        cur_time = (end - start)/(double) CLOCKS_PER_SEC;
        cout << "total time: " << avg_time << endl;
    }

    return fhp;
    /*
    stringstream ss;
    ss << "estimated_fhp/" << config.graph_alias << "/samba/pairwise/" << target << ".txt";
    string outfile = ss.str();
    ofstream est_ppr_file(outfile);

    est_ppr_file << source<<" "<< target << " " << dht << endl;

    est_ppr_file.close();
     */
}

void fhp_power_iteration(const Graph& graph, int source, vector<int> t_set, double& fhp, double& time){
    clock_t start = clock();

    unordered_map<int, double> map_residual(graph.n);
    map_residual[source] = 1.0;

    int num_iter=0;
    fhp = 0.0;
    while( num_iter < config.max_iter_num )
    {
        if (num_iter % 10 == 0)
        {
            cout << "Iter No. : " << num_iter << endl;
        }
        num_iter++;
        // INFO(num_iter, rsum);
        for(int i=0; i<t_set.size(); i++){
            int temp_target = t_set[i];
            fhp += map_residual[temp_target];
            map_residual[temp_target] = 0.0;
        }

        vector< pair<int,double>> pairs(map_residual.begin(), map_residual.end());
        map_residual.clear();

        for(const auto &p: pairs){
            if(p.second > 0){
                int out_deg = graph.g[p.first].size();

                double remain_residual = (1-config.alpha)*p.second;

                if(out_deg==0)
                {
                    continue;
                }

                double avg_push_residual = remain_residual / out_deg;
                for (int next : graph.g[p.first]) {
                    map_residual[next] += avg_push_residual;
                }
            }
        }

        pairs.clear();
    }
    map_residual.clear();

    clock_t end = clock();
    avg_time += (end - start)/(double) CLOCKS_PER_SEC;
    time = (end - start)/(double) CLOCKS_PER_SEC;
    cout << "total time: " << avg_time << endl;
}


static void backward_push_for_topk_group_balance(int s, vector<int>& t_vec, const Graph &graph, double R_max,
                    unordered_map<int, double>& res_map, double &reserve)
{
    // bwd_idx.first.clean();
    // bwd_idx.second.clean();

    static unordered_map<int, bool> idx;
    idx.clear();

    unordered_set<int> t_set(t_vec.begin(), t_vec.end());

    vector<int> q;
    q.reserve(graph.n);
    q.push_back(-1);
    unsigned long left = 1;

    double myeps =1.0/t_vec.size(); //10^{-10}

    for (int target: t_vec)
    {
        //bwd_idx.second.insert(target, 1.0);
        res_map[target] = 1.0;
        q.push_back(target);
        idx[target] = true;
    }

    // bwd_idx.second.insert(s, 0);
    // bwd_idx.first.insert(s, 0);//s always have reserve by converting its residues

    double remove_residual = 0.0;
    long long push_operation = 0;
    int iter = 0;
    //double source_residual = 0.0;
    while(R_max <= myeps)
     {
        cout << "iteration " << iter << ", eps: " << myeps <<endl;
        //t is not an out-neighbor of s
        while (left < q.size()) {
            int v = q[left];
            idx[v] = false;
            left++;

            // if (bwd_idx.second[v] < myeps)
            if (res_map[v] < myeps)
                continue;

            if(v == s){//
                // bwd_idx.first[s] += bwd_idx.second[v];
                reserve += res_map[v];
            }

            //double residual = (1.0 - config.alpha) * bwd_idx.second[v];
            double residual = (1.0 - config.alpha) * res_map[v];

            // bwd_idx.second[v] = 0;
            res_map.erase(v);
            if (graph.gr[v].size() > 0)
            {
                for (int next : graph.gr[v])
                {

                    double incr = residual/graph.g[next].size();

                    if (t_set.find(next) != t_set.end())
                    {
                        remove_residual += incr;
                        continue;
                    }

                    push_operation++;

                    // if (bwd_idx.second.notexist(next))
                    //     bwd_idx.second.insert(next, incr);
                    // else
                    //     bwd_idx.second[next] += incr;
                    res_map[next] += incr;

                    //if ( bwd_idx.second[next] > myeps ) {
                    if ( res_map[next] > myeps ) {

                        /*
                        if (next == s || idx[next] == true)
                        {
                            continue;
                        }*/
                        if(idx[next] == true){
                            continue;
                        }
                        // put next into q if next is not in q
                            idx[next] = true;//(int) q.size();
                            q.push_back(next);
                    }
                }
            }
        }

        // 防止source被push
        //source_residual += bwd_idx.second[s];
        //bwd_idx.second[s] = 0.0;

        if (myeps <= R_max)
        {
            break;
        }

        bool time_stop = false;
        balance_rmax(myeps, push_operation, time_stop);
        if (time_stop)
        {
            break;
        }

        myeps = myeps/2.0;
        myeps = (R_max > myeps)?R_max:myeps;
        //double percent = (double)(bwd_idx.second.occur.m_num)/graph.n;
        // double percent = (double)(res_map.size())/graph.n;
        // cout << "occur percent: " << percent << endl;
        // if (percent < 0.1)
        // {
        //     for (int j=0; j<bwd_idx.second.occur.m_num; j++)
        //     {
        //         int node = bwd_idx.second.occur[j];
        //         if (bwd_idx.second[node] > myeps)
        //         {
        //             q.push_back(node);
        //             idx[node] = true;
        //         }
        //     }
        // }
        // else
        // {
        //     for (int j = 0; j < graph.n; j++)
        //     {
        //         if (bwd_idx.second.exist(j) && bwd_idx.second[j] > myeps)
        //         {
        //             q.push_back(j);
        //             idx[j] = true;
        //         }
        //     }
        // }

        for (auto& elem: res_map)
        {
            if (elem.second > myeps)
            {
                q.push_back(elem.first);
                idx[elem.first] = true;
            }
        }
    }

    //bwd_idx.second[s] = source_residual;
}

static double test_time = 0.0;
static long long push_operation = 0;
void print_test_time()
{
    cout << "foreach time: " << test_time << endl;
    cout << "push_operation: " << push_operation << endl;
    push_operation = 0;
    test_time = 0.0;
}


static void backward_push_for_topk_group(int s, vector<int>& t_vec, const Graph &graph, double R_max,
                    unordered_map<int, double>& res_map, double& reserve)
{
    // bwd_idx.first.clean();
    // bwd_idx.second.clean();

    // static unordered_map<int, bool> idx;
    // idx.clear();

    unordered_map<int, bool> idx;

    unordered_set<int> t_set(t_vec.begin(), t_vec.end());

    vector<int> q;
    //q.reserve(graph.n);
    q.push_back(-1);
    unsigned long left = 1;

    double myeps = R_max; //10^{-10}

    clock_t for_start = clock();
    for (auto &elem: res_map)
    {
        //bwd_idx.second.insert(target, 1.0);
        if (elem.second > R_max)
        {
            q.push_back(elem.first);
            idx[elem.first] = true;
        }

    }
    clock_t for_end = clock();
    test_time = (for_end - for_start)/(double)CLOCKS_PER_SEC;

    double remove_residual = 0.0;

    if(1)
    {
        //t is not an out-neighbor of s
        while (left < q.size()) {
            int v = q[left];
            idx[v] = false;
            left++;
            // if (bwd_idx.second[v] < myeps)
            if (res_map[v] < myeps)
                continue;

            if(v == s)
            {//
                // bwd_idx.first[s] += bwd_idx.second[v];
                reserve += res_map[v];
            }

            //double residual = (1.0 - config.alpha) * bwd_idx.second[v];
            double residual = (1.0 - config.alpha) * res_map[v];

            //bwd_idx.second[v] = 0;
            res_map.erase(v);

            if (graph.gr[v].size() > 0)
            {
                for (int next : graph.gr[v])
                {

                    double incr = residual/graph.g[next].size();

                    if (t_set.find(next) != t_set.end())
                    {
                        remove_residual += incr;
                        continue;
                    }
                    push_operation++;
                    // if (bwd_idx.second.notexist(next))
                    //     bwd_idx.second.insert(next, incr);
                    // else
                    //     bwd_idx.second[next] += incr;
                    res_map[next] += incr;

                    // if (next != s && bwd_idx.second[next] > myeps && idx[next] != true) {
                    if ( res_map[next] > myeps )
                    {

                            if(idx[next] == true)
                            {
                                continue;
                            }
                        // put next into q if next is not in q
                            idx[next] = true;//(int) q.size();
                            q.push_back(next);
                    }

                }
            }
        }

        // for (int target: t_vec)
        // {
        //     //bwd_idx.second.insert(target, 1.0);
        //     if (res_map[target] != 0.0)
        //     {
        //         cout << "target " << target << ", res value: " << res_map[target] << endl;
        //     }
        // }
    }
    else{
       cout<<"t is an out-neighbor of s such that no backward search is performed" <<endl;
    }

    // cout << "remove_residual: " << remove_residual << endl;
    // cout << "push_operation: " << push_operation <<endl;

}

double samba_query_group_sqrt_walk(int source, vector<int> t_vec, const Graph& graph, double& cur_time) {
    //Timer timer(BIPPR_QUERY);
    //这里需要重新设置config.omega和rmax
    samba_target_group_setting(graph.n, graph.m, config.gsize);

    clock_t start = clock();
    double fhp = 0.0;
    //double stop_prob =config.alpha;
    //double sqrt_value = 1;

    double sqrt_value = sqrt(1-config.alpha);
    double stop_prob = 1 - sqrt_value;
    cout<<"rev stop probability: " << stop_prob <<endl;

    unordered_set<int> t_set(t_vec.begin(), t_vec.end());
    sqrt_rw_counter.clean();
    if (t_set.find(source) != t_set.end()) {
        fhp = 1.0;
    }
    else{
        //double new_rmax = 0.001; //10^{-6}
        INFO(config.rmax);
        //cout<< "rmax used: " << new_rmax <<endl;
        //first run the backward propagation from the target
        reverse_local_update_linear_visiting_prob_group(source, t_vec, graph, config.rmax);

        cout << "time for linear update time: " << (clock() - start)/(double) CLOCKS_PER_SEC << endl;
        //simulate the random walks from the source

        INFO(config.omega);
        double total_X = 0.0;
        unsigned long total_visits = 0;

        const double incre_unit = sqrt_value;
        sqrt_rw_counter.insert(source, 0);
        //rw_counter.clean();
        for (unsigned long j = 0; j < config.omega; j++) {
            //a random walk from i
            //terminates when it hits t
            int temp_node = source;
            int step = 0;
            sqrt_rw_counter[source] += 1.0;
            double incre = 1.0;
            while (drand() > stop_prob)
            {
                step += 1;
                incre *= incre_unit;
                if (graph.g[temp_node].size()) {
                    int next = lrand() % graph.g[temp_node].size();
                    temp_node = graph.g[temp_node][next];

                    if(t_set.find(temp_node) != t_set.end()){
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
        //add the reserve of node source
        fhp += bwd_idx.first[source];

        for(long j=0; j<bwd_idx.second.occur.m_num; j++){
            // ppr[i]+=counts[residue.first]*1.0/config.omega*residue.second;
            int nodeid = bwd_idx.second.occur[j];
            double residual = bwd_idx.second[nodeid];
            if(t_set.find(nodeid) != t_set.end()) {
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

    clock_t end = clock();
    avg_time += (end - start)/(double) CLOCKS_PER_SEC;
    cur_time = (end - start)/(double) CLOCKS_PER_SEC;

    cout<< "final fhp: " <<fhp <<endl;
    cout<< "total time: " << avg_time << endl;
    return fhp;
    /*
    stringstream ss;
    ss << "estimated_fhp/" << config.graph_alias << "/samba/pairwise/" << target << ".txt";
    string outfile = ss.str();
    ofstream est_ppr_file(outfile);

    est_ppr_file << source<<" "<< target << " " << dht << endl;

    est_ppr_file.close();
     */
}


#endif