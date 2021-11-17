
#ifndef BUILD_H
#define BUILD_H

#include "algo.h"
#include "graph.h"
#include "heap.h"
#include "config.h"
#include <time.h>

#include <boost/serialization/serialization.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/utility.hpp>


inline string get_hub_fwd_idx_file_name(){
    string prefix = config.prefix + FILESEP + config.graph_alias+FILESEP;
    prefix += config.graph_alias + ".eps-" + to_str(config.epsilon);
    // prefix += ".space-1";
    prefix += ".space-" + to_str(config.hub_space_consum);

    string suffix;

    suffix += ".compress.fwd.idx";
    string file_name = prefix + suffix;
    return file_name;
}

inline string get_hub_fwd_idx_info_file_name(){
    string idx_file = get_hub_fwd_idx_file_name();
    return replace(idx_file, "fwd.idx", "fwd.info");
}

inline string get_hub_bwd_idx_file_name(){
    string idx_file = get_hub_fwd_idx_file_name();
    return replace(idx_file, "fwd.idx", "bwd.idx");
}

inline void deserialize_hub_fwd_idx(){
    string file_name = get_hub_fwd_idx_file_name();
    assert_file_exist("index file", file_name);
    std::ifstream ifs(file_name);
    boost::archive::binary_iarchive ia(ifs);
    ia >> hub_fwd_idx;
    
    string rwn_file_name = get_hub_fwd_idx_file_name()+".rwn";
    assert_file_exist("rwn file", rwn_file_name);
    std::ifstream ofs_rwn(rwn_file_name);
    boost::archive::binary_iarchive ia_rwn(ofs_rwn);

    ia_rwn >> hub_sample_number;


    string info_file = get_hub_fwd_idx_info_file_name();
    assert_file_exist("info file", info_file);
    std::ifstream info_ofs(info_file);
    boost::archive::binary_iarchive info_ia(info_ofs);
    info_ia >> hub_fwd_idx_cp_pointers;
}

inline void deserialize_hub_bwd_idx(){
    string file_name = get_hub_bwd_idx_file_name();
    // assert_file_exist("index file", file_name);
    if (!exists_test(file_name)) {
        cerr << "index file " << file_name << " not find " << endl;
        return;
    }
    std::ifstream ifs(file_name);
    boost::archive::binary_iarchive ia(ifs);
    ia >> hub_bwd_idx;
}

void load_hubppr_oracle(const Graph& graph){
    deserialize_hub_fwd_idx();

    // fwd_idx_size.initialize(graph.n);
    hub_fwd_idx_ptrs.resize(graph.n);
    hub_fwd_idx_size.resize(graph.n);
    std::fill(hub_fwd_idx_size.begin(), hub_fwd_idx_size.end(), 0);
    hub_fwd_idx_size_k.initialize(graph.n);


    for(auto &ptrs: hub_fwd_idx_cp_pointers){
        int node = ptrs.first;
        int size=0;

        unsigned long long ptr = ptrs.second[0];
        unsigned long long end_ptr = ptrs.second[ptrs.second.size()-1];
        for(; ptr<end_ptr; ptr+=2){
            size += hub_fwd_idx[ptr+1];
        }

        hub_fwd_idx_ptrs[node] = ptrs.second;

        // fwd_idx_size.insert(node, size);
        hub_fwd_idx_size[node] = size;

        int u = 1 + floor(log( hub_fwd_idx_size[node]*1.0 )/log(2)); //we can pre-compute to avoid reduplicate computation
        int k = pow(2, u-1)-1;
        hub_fwd_idx_size_k.insert(node, k);
    }

    hub_fwd_idx_cp_pointers.clear();

    INFO(hub_fwd_idx_size.size());

    deserialize_hub_bwd_idx();
    INFO(hub_bwd_idx.size());
}

inline string get_exact_topk_ppr_file(){
    if(!boost::algorithm::ends_with(config.exact_pprs_folder, FILESEP))
        config.exact_pprs_folder += FILESEP;
    return config.exact_pprs_folder+config.graph_alias+".topk.pprs";
}

inline void save_exact_topk_ppr(){
    string filename = get_exact_topk_ppr_file();
    std::ofstream ofs(filename);
    boost::archive::text_oarchive oa(ofs);
    oa << exact_topk_pprs;
}

inline void load_exact_topk_ppr(){
    string filename = get_exact_topk_ppr_file();
    if(!exists_test(filename)){
        INFO("No exact topk ppr file", filename);
        return;
    }
    std::ifstream ifs(filename);
    boost::archive::text_iarchive ia(ifs);
    ia >> exact_topk_pprs;

    INFO(exact_topk_pprs.size());
}
inline vector<int> load_estimated_topk_fhp(int source, int k, int n, string method){
    vector<int> realList;
    stringstream ss;

    if(method == "samba"){
        ss << "estimated_fhp/" << config.graph_alias << "/topk/"<<method<<"/"<<k<<"/"<<to_str(source)  <<".txt";
        string infile = ss.str();
        ifstream topk_file(infile);

        vector<double> simList;
        for(int i = 0; i < k; i++){
            int tempId;
            double tempSim;
            topk_file >> tempId >> tempSim;
            if(i >= k && tempSim < simList[k-1]){
                break;
            }//if( i == 0)
            //    continue;
            realList.push_back(tempId);
            simList.push_back(tempSim);
        }
        topk_file.close();
    }else {
        ss << "estimated_fhp/" << config.graph_alias << "/topk/" << method << "/" << to_str(source) << ".txt";
        string infile = ss.str();
        ifstream topk_file(infile);
       // vector<int> realList;
        vector<double> simList;
        for(int i = 0; i < k; i++){
            int tempId;
            double tempSim;
            topk_file >> tempId >> tempSim;
            if(i >= k && tempSim < simList[k-1]){
                break;
            }
            //if( i == 0)
            //    continue;
            realList.push_back(tempId);
            simList.push_back(tempSim);
        }
        topk_file.close();
    }

    return realList;
}

//取s点的groundtruth
inline vector<int> load_exact_topk_fhp(int source, int k, int n){
    stringstream ss;
    ss << "real_fhp/" << config.graph_alias << "/mc_results_2/"<<to_str(source)  <<".txt";
    string infile = ss.str();
    ifstream topk_file(infile);
    vector<int> realList;
    vector<double> simList;
    for(int i = 0; i < n; i++){
        int tempId;
        double tempSim;
        topk_file >> tempId >> tempSim;
        if(i >= k && tempSim < simList[k-1]){
            break;
        }
        //if( i == 0)
        //    continue;
        realList.push_back(tempId);
        simList.push_back(tempSim);
    }
    topk_file.close();
    return realList;
}

inline unordered_map<int, double> load_exact_topk_map_fhp(int source, int k, int n){
    unordered_map<int, double> answer_map;
    stringstream ss;
    ss << "real_fhp/" << config.graph_alias << "/mc_results/"<<to_str(source)  <<".txt";
    string infile = ss.str();
    ifstream topk_file(infile);
    double k_Sim = 0;
    for(int i = 0; i < n; i++){
        int tempId;
        double tempSim;
        topk_file >> tempId >> tempSim;
        if(i == k - 1){
            k_Sim = tempSim;
        }
        if(i >= k && tempSim < k_Sim){
            break;
        }
        answer_map[tempId] = tempSim;
    }
    topk_file.close();
    return answer_map;
}



inline string get_idx_file_name(){
    string file_name;
    if(config.rmax_scale==1)
        file_name = config.graph_location+"randwalks_"+to_string(config.epsilon)+".idx";
    else
        file_name = config.graph_location+"randwalks."+to_string(config.rmax_scale)+".idx";
    
    return file_name;
}

inline string get_idx_info_name(){
    string file_name;
    if(config.rmax_scale==1)
        file_name = config.graph_location+"randwalks_"+to_string(config.epsilon)+"..info";
    else
        file_name = config.graph_location+"randwalks."+to_string(config.rmax_scale)+".info";
    return file_name;   
}

inline void deserialize_idx(){
    string file_name = get_idx_file_name();
    assert_file_exist("index file", file_name);
    std::ifstream ifs(file_name);
    boost::archive::binary_iarchive ia(ifs);
    ia >> rw_idx;

    file_name = get_idx_info_name();
    assert_file_exist("index file", file_name);
    std::ifstream info_ifs(file_name);
    boost::archive::binary_iarchive info_ia(info_ifs);
    info_ia >> rw_idx_info;
}

inline void serialize_idx(){
    std::ofstream ofs(get_idx_file_name());
    boost::archive::binary_oarchive oa(ofs);
    oa << rw_idx;

    std::ofstream info_ofs(get_idx_info_name());
    boost::archive::binary_oarchive info_oa(info_ofs);
    info_oa << rw_idx_info;
}

double calNDCG(vector<int> candidates, int k, int s, int n){
    vector<int> topK = load_exact_topk_fhp(s, k, n);
    unordered_map<int, double> realMap = load_exact_topk_map_fhp(s, k, n);

   /* double correct = 0;
    for(int i = 0; i < k; i++){
        if(realMap[candidates[i]] == realMap[topK[i]])
            correct++;
        else{
            cout << "misMatch : " << candidates[i] << ", " << topK[i] << endl;
        }
    }
    return correct / (double)k;
*/
    double Zp = 0;
    for(int i = 1; i <= k; i++){
        Zp += (pow(2, realMap[topK[i-1]]) - 1) / (log(i+1) / log(2));
    }
    double NDCG = 0;
    for(int i = 1; i <= k; i++){
        NDCG += (pow(2, realMap[candidates[i-1]]) - 1) / (log(i+1) / log(2));
    }
    return NDCG / Zp;
}
/*
double calPrecision(vector<int> topK1, vector<int> realList, int k, bool isShowMissing = false){
    int size = realList.size();
    //cout<<size<<endl;
    int size2 = topK1.size();
    //cout<<size2<<endl;
    int hitCount = 0;
    for(int i = 0; i < size2; i++) {
        //cout<<i<<" " << topK1[i]<<" ";
        bool isFind = false;
        for (int j = 0; j < size; j++) {
            //cout<< realList[j] <<endl;
            if (topK1[i] == realList[j]) {
                //cout<< topK1[i] <<" " << realList[j] <<endl;
                hitCount++;
                isFind = true;
                break;
            }
        }
        if(!isFind){
           cout << "useless node: " << topK1[i] << endl;
        }
    }
    cout << "hit Count: " << hitCount << endl;
    double result = hitCount / (double) k;
    return result < 1 ? result : 1;

}
*/

void single_build(const Graph& graph, int start, int end, vector<int>& rw_data, unordered_map<int, pair<unsigned long long, unsigned long> >& rw_info_map, int core_id){
    unsigned long num_rw;
    for(int v=start; v<end; v++){
        num_rw = ceil(graph.g[v].size()*config.rmax*config.omega);
        rw_info_map[v] = MP(rw_data.size(), num_rw);
        for(unsigned long i=0; i<num_rw; i++){
            int des = random_walk_thd(v, graph, core_id);
            rw_data.push_back(des);
        }
    }
}

void multi_build(const Graph& graph){
    INFO("multithread building...");
    fora_setting(graph.n, graph.m);

    // rw_idx = RwIdx( graph.n, vector<int>() );
    rw_idx_info.resize(graph.n);

    unsigned NUM_CORES = std::thread::hardware_concurrency();
    assert(NUM_CORES >= 2);

    INFO(NUM_CORES);

    unsigned long long rw_max_size = graph.m*config.rmax*config.omega;
    INFO(rw_max_size, rw_idx.max_size());

    rw_idx.reserve(rw_max_size);

    vector< vector<int> > vec_rw(NUM_CORES+1);
    vector< unordered_map<int, pair<unsigned long long, unsigned long> > > vec_rw_info(NUM_CORES+1);
    std::vector< std::future<void> > futures(NUM_CORES+1);

    int num_node_per_core = graph.n/(NUM_CORES+1);
    int start=0;
    int end=0;

    {
        INFO("rand-walking...");
        Timer tm(1);
        for(int core_id=0; core_id<NUM_CORES+1; core_id++){
            end = start + num_node_per_core;
            if(core_id==NUM_CORES)
                end = graph.n;
            
            vec_rw[core_id].reserve(rw_max_size/NUM_CORES);
            futures[core_id] = std::async( std::launch::async, single_build, std::ref(graph), start, end, std::ref(vec_rw[core_id]), std::ref(vec_rw_info[core_id]), core_id );
            start = end;
        }
        std::for_each( futures.begin(), futures.end(), std::mem_fn(&std::future<void>::wait));
    }

    {
        INFO("merging...");
        Timer tm(2);
        start=0;
        end=0;
        for(int core_id=0; core_id<NUM_CORES+1; core_id++){
            end = start + num_node_per_core;
            if(core_id==NUM_CORES)
                end = graph.n;

            rw_idx.insert( rw_idx.end(), vec_rw[core_id].begin(), vec_rw[core_id].end() );

            for(int v=start; v<end; v++){
                unsigned long long p = vec_rw_info[core_id][v].first;
                unsigned long num_rw = vec_rw_info[core_id][v].second;
                rw_idx_info[v] = MP( p + rw_idx.size()-vec_rw[core_id].size(), num_rw);
            }
            start = end;
        }
    }

    {
        INFO("materializing...");
        INFO(rw_idx.size(), rw_idx_info.size());
        Timer tm(3);
        serialize_idx(); //serialize the idx to disk
    }

    cout << "Memory usage (MB):" << get_proc_memory()/1000.0 << endl << endl;
}

void build(const Graph& graph){
    // size_t space = get_ram_size();
    // size_t estimated_space = sizeof(RwIdx) + graph.n *( sizeof(vector<int>) + config.num_rw*sizeof(int) );

    // if(estimated_space > space) //if estimated raw space overflows system maximum raw space, reset number of rand-walks
    //     config.num_rw = space * config.num_rw / estimated_space;
    clock_t startTime = clock();

    fora_setting(graph.n, graph.m);

    // rw_idx = RwIdx( graph.n, vector<int>() );
    rw_idx_info.resize(graph.n);

    unsigned long long rw_max_size = graph.m*config.rmax*config.omega;
    INFO(rw_max_size, rw_idx.max_size());

    rw_idx.reserve(rw_max_size);

    {
        INFO("rand-walking...");
        Timer tm(1);
        unsigned long num_rw;
        for(int source=0; source<graph.n; source++){ //from each node, do rand-walks
            num_rw = ceil(graph.g[source].size()*config.rmax*config.omega);
            rw_idx_info[source] = MP(rw_idx.size(), num_rw);
            for(unsigned long i=0; i<num_rw; i++){ //for each node, do some rand-walks
                int destination = random_walk(source, graph);
                // rw_idx[source].push_back(destination);
                rw_idx.push_back(destination);
            }
        }
    }

    {
        INFO("materializing...");
        INFO(rw_idx.size(), rw_idx_info.size());
        Timer tm(2);
        serialize_idx(); //serialize the idx to disk
    }

    clock_t endTime = clock();
    double totalTime = (endTime - startTime) / CLOCKS_PER_SEC;
    cout << "time cost: " << totalTime <<endl;
    cout << "Memory usage (MB):" << get_proc_memory()/1000.0 << endl << endl;
    stringstream ss;
    ss << "data/" << config.graph_alias << "/pre_time_"<<config.epsilon<<".txt";
    string outfile = ss.str();
    ofstream pre_time_file(outfile);
    pre_time_file << "epsilon: " << config.epsilon <<endl;
    pre_time_file << "pre_time: " << totalTime <<endl;
    pre_time_file << "Memory usage (MB):" << get_proc_memory()/1000.0 << endl;
    pre_time_file.close();
}
void build_dynamic_for_each_node(const Graph& graph){
    fora_setting(graph.n, graph.m);

    // rw_idx = RwIdx( graph.n, vector<int>() );
    rw_idx_info.resize(graph.n);

    unsigned long long rw_max_size = graph.m*config.rmax*config.omega;
    INFO(rw_max_size, rw_idx.max_size());

    rw_idx.reserve(rw_max_size);

    {
        INFO("rand-walking...");
        Timer tm(1);
        unsigned long num_rw;
        for(int source=0; source<graph.n; source++){ //from each node, do rand-walks
            num_rw = ceil(graph.g[source].size()*config.rmax*config.omega);
            rw_idx_info[source] = MP(rw_idx.size(), num_rw);
            for(unsigned long i=0; i<num_rw; i++){ //for each node, do some rand-walks
                int destination = random_walk(source, graph);
                // rw_idx[source].push_back(destination);
                rw_idx.push_back(destination);
            }
        }
    }

    {
        INFO("materializing...");
        INFO(rw_idx.size(), rw_idx_info.size());
        //Timer tm(2);
        //serialize_idx(); //serialize the idx to disk
    }

}

void build_dynamic(Graph& graph){
    // load the nodes for dynamically removing
    vector<int> dynamic_nodes;
    load_nodes_for_dynamic(dynamic_nodes);
    int num_dynamic_nodes = dynamic_nodes.size();

    double totalTime = 0.0;
    //clock_t startTime = clock();

    for(int i=0; i< num_dynamic_nodes; i++){
        cout<<"processing node: " << i+1 <<endl;
        int tempNode = dynamic_nodes[i];
        cout<<"node: " <<tempNode<<endl;
        int outdeg_tempNode = graph.g[tempNode].size();
        //cout<<"out degree: " << outdeg_tempNode<<endl;
        graph.g[tempNode].clear();
        graph.m = graph.m  - outdeg_tempNode;
        int indeg_tempNode = graph.gr[tempNode].size();
        //cout<<"in degree: " << indeg_tempNode<<endl;
        //graph.gr[tempNode].clear();
        for(int j = 0; j < indeg_tempNode; j++){
            int in_neighbor = graph.gr[tempNode][j];
            //cout<<"in neighbor: " << in_neighbor <<endl;
            //cout<< graph.g[in_neighbor].size() <<endl;
            if(graph.g[in_neighbor].size() != 0)
            {
                for(auto it = graph.g[in_neighbor].begin(); it != graph.g[in_neighbor].end(); ++it){
                    //cout<<*it<<endl;
                    if(*it == tempNode){
                        graph.g[in_neighbor].erase(it);
                        //cout<< graph.g[in_neighbor].size() <<endl;
                        break;
                    }
                }
            }
        }
        graph.m = graph.m - indeg_tempNode;
        cout<<"m: " << graph.m <<endl;
        graph.n = graph.n -1;
        cout<<"n: " << graph.n <<endl;

        clock_t startTime = clock();
        build_dynamic_for_each_node(graph);
        clock_t endTime = clock();
        totalTime += (endTime - startTime) /(double) CLOCKS_PER_SEC;
    }
    //clock_t endTime = clock();
    //double totalTime = (endTime - startTime) / CLOCKS_PER_SEC;
    cout << "time cost: " << totalTime / (double) num_dynamic_nodes <<endl;
    cout << "Memory usage (MB):" << get_proc_memory()/1000.0 << endl << endl;
    stringstream ss;
    ss << "data/" << config.graph_alias << "/pre_time_for_dynamic.txt";
    string outfile = ss.str();
    ofstream pre_time_file(outfile);
    pre_time_file << "epsilon: " << config.epsilon <<endl;
    pre_time_file << "pre_time: " << totalTime / (double) num_dynamic_nodes <<endl;
    pre_time_file << "Memory usage (MB):" << get_proc_memory()/1000.0 << endl;
    pre_time_file.close();
}



#endif
