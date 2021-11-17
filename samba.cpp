
#define _CRT_SECURE_NO_DEPRECATE
#define HEAD_INFO

#include "mylib.h"
#include "graph.h"
#include "algo.h"
#include "query.h"
#include "group.h"

#include <boost/progress.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>

#include <chrono>


using namespace std::chrono;

using namespace boost;
using namespace boost::property_tree;

using namespace std;


string get_time_path() {
    using namespace boost::posix_time;
    auto tm = second_clock::local_time();
#ifdef WIN32
    return  "../../execution/" + to_iso_string(tm);
#else
    return parent_folder+FILESEP+"execution/" + to_iso_string(tm);
#endif
}

#include <boost/program_options.hpp>

namespace po = boost::program_options;


using namespace std;

int main(int argc, char *argv[]) {
    ios::sync_with_stdio(false);
    program_start(argc, argv);

    // this is main entry
    Saver::init();
    srand(time(NULL));
    config.graph_alias = "nethept";
    for (int i = 0; i < argc; i++) {
        string help_str = ""
                "samba query --algo <algo> [options]\n"
                "samba generate-ss-query [options]\n"
                "samba gen-exact-topk [options]\n"
                "samba\n"
                "\n"
                "algo: \n"
                "  samba\n"
                "  samba_topk\n"
                "  mc\n"
                "  mc_sqrt\n"
                "options: \n"
                "  --prefix <prefix>\n"
                "  --epsilon <epsilon>\n"
                "  --dataset <dataset>\n"
                "  --query_size <queries count>\n"
                "  --k <top k>\n"
                "  --with_idx\n"
                "  --rw_ratio <rand-walk cost ratio>\n";

        if (string(argv[i]) == "--help") {
            cout << help_str << endl;
            exit(0);
        }
    }

    config.action = argv[1];
    cout << "action: " << config.action << endl;

    // init graph first
    for (int i = 0; i < argc; i++) {
        if (string(argv[i]) == "--prefix") {
            config.prefix = argv[i + 1];
        }
        if (string(argv[i]) == "--dataset") {
            config.graph_alias = argv[i + 1];
        }
    }

    for (int i = 0; i < argc; i++) {
        string arg = argv[i];
        if (arg == "--algo") {
    		config.algo = string(argv[i + 1]);
        }

        else if (arg == "--epsilon") {
            config.epsilon = atof(argv[i + 1]);
            INFO(config.epsilon);
        }else if(arg == "--multithread"){
             config.multithread = true;
        }
        else if(arg == "--result_dir"){
            config.exe_result_dir = string(argv[i + 1]);
        }
        else if( arg == "--exact_ppr_path"){
            config.exact_pprs_folder = string(argv[i + 1]);
        }
        else if(arg == "--with_idx"){
            config.with_rw_idx = true;
        }
        else if(arg == "--rmax") {
            config.rmax = atof(argv[i + 1]);
        }
        else if (arg == "--query_size"){
            config.query_size = atoi(argv[i+1]);
        }else if (arg == "--group_num"){
            config.group_num=atoi(argv[i+1]);
        }else if(arg == "--hub_space"){
            config.hub_space_consum = atoi(argv[i+1]);
        }else if (arg == "--version"){
            config.version = argv[i+1];
        }
        else if(arg == "--k"){
            config.k = atoi(argv[i+1]);
        }
        else if(arg == "--num_rw") {
            config.num_rw = atoi(argv[i + 1]);
        }
        else if (arg == "--prefix" || arg == "--dataset") {
            // pass
        }
        else if (arg == "--gsize")
        {
            config.gsize =atoi(argv[i+1]);
        }
        else if (arg == "--cluster")
        {
            config.is_cluster = (atoi(argv[i+1]) == 1);
        }
        else if (arg == "--sqrt_walk")
        {
            config.sqrt_walk = (atoi(argv[i+1]) == 1);
        }

        else if (arg.substr(0, 2) == "--") {
            cerr << "command not recognize " << arg << endl;
            exit(1);
        }
    }

    INFO(config.version);
    vector<string> possibleAlgo = {MC_TOPK, MC_TOPK_GROUP, FAST_PRUNE, FAST_PRUNE_GROUP, MC, MC_GROUP, SAMBA, FORWARD, POWER};

    INFO(config.action);

    srand (time(NULL));
    if (config.action == QUERY) {
        auto f = find(possibleAlgo.begin(), possibleAlgo.end(), config.algo);
        assert (f != possibleAlgo.end());
        if(f == possibleAlgo.end()){
            INFO("Wrong algo param: ", config.algo);
            exit(1);
        }

        config.graph_location = config.get_graph_folder();
        Graph graph(config.graph_location);
        INFO("load graph finish");
        init_parameter(config, graph);

        INFO("finihsed initing parameters");
        INFO(graph.n, graph.m);

        //if(config.with_rw_idx)
        //    deserialize_idx();

        query(graph);
    }else if(config.action == TOPK){
        auto f = find(possibleAlgo.begin(), possibleAlgo.end(), config.algo);
        assert (f != possibleAlgo.end());
        if(f == possibleAlgo.end()){
            INFO("Wrong algo param: ", config.algo);
            exit(1);
        }

        config.graph_location = config.get_graph_folder();
        Graph graph(config.graph_location);
        INFO("load graph finish");
        init_parameter(config, graph);

        INFO("finihsed initing parameters");
        INFO(graph.n, graph.m);

        //if(config.with_rw_idx)
        //    deserialize_idx();

        topk_query(graph);
    }
    else if (config.action == GEN_SS_QUERY) {
        config.graph_location = config.get_graph_folder();
        Graph graph(config.graph_location);
        INFO(graph.n, graph.m);

        generate_ss_query(graph.n);
    }
    else if (config.action == GEN_GROUP)
    {
        config.graph_location = config.get_graph_folder();
        Graph graph(config.graph_location);
        INFO(graph.n, graph.m);
        generate_group(graph.n, graph, config.gsize, config.is_cluster);
    }
    else if (config.action == GEN_EXACT_TOPK){
        config.graph_location = config.get_graph_folder();
        Graph graph(config.graph_location);
        INFO("load graph finish");
        init_parameter(config, graph);

        if(config.exact_pprs_folder=="" || !exists_test(config.exact_pprs_folder))
            config.exact_pprs_folder = config.graph_location;

        INFO("finihsed initing parameters");
        INFO(graph.n, graph.m);

        multi_mc_topk_fhp(graph);
    }
    else {
        cerr << "sub command not regoznized" << endl;
        exit(1);
    }

    Timer::show();
    if(config.action == QUERY || config.action == TOPK) {
        Counter::show();
        auto args = combine_args(argc, argv);
        Saver::save_json(config, result, args);
    }

    program_stop();
    return 0;
}
