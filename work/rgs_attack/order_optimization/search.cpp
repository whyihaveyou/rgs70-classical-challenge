#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>
using Bits=unsigned __int128;
using Clock=std::chrono::steady_clock;
constexpr int N=70;
std::array<Bits,N> adj;
const Bits full=(Bits(1)<<N)-1;
int msb(Bits x){uint64_t hi=x>>64;return hi?127-__builtin_clzll(hi):63-__builtin_clzll(uint64_t(x));}
int rank_cut(Bits left){
  if(__builtin_popcountll(uint64_t(left))+__builtin_popcountll(uint64_t(left>>64))>N/2)left=full^left;
  Bits right=full^left, todo=left;
  std::array<Bits,N> piv{};int rank=0;
  while(todo){int v=msb(todo);todo^=Bits(1)<<v;Bits x=adj[v]&right;
    while(x){int p=msb(x);if(piv[p])x^=piv[p];else{piv[p]=x;++rank;break;}}
  }return rank;
}
struct State{std::array<int,N> order,ranks;std::array<Bits,N> prefix;uint64_t cost;};
void update(State &s,int lo=0,int hi=N-1){
  Bits p=lo?s.prefix[lo-1]:0;
  for(int k=lo;k<=hi;++k){p|=Bits(1)<<s.order[k];s.prefix[k]=p;}
  for(int k=lo;k<hi;++k)s.ranks[k]=rank_cut(s.prefix[k]);
  if(lo==0&&hi==N-1)s.ranks[N-1]=0;
  s.cost=0;for(int r:s.ranks)s.cost+=uint64_t(1)<<r;
}
int peak(const State&s){return *std::max_element(s.ranks.begin(),s.ranks.end());}
void save(const State&s,const std::string&f){std::ofstream o(f);for(int v:s.order)o<<v<<' ';o<<'\n';}
int main(int argc,char**argv){
  if(argc!=4)return 2;std::ifstream in(argv[1]);State start;
  for(int i=0;i<N;++i)for(int j=0;j<N;++j){int e;in>>e;if(e)adj[i]|=Bits(1)<<j;}
  for(auto&v:start.order)in>>v;update(start);State best=start,cur=start;
  const double budget=std::stod(argv[2]);std::mt19937_64 rng(20261030);auto t0=Clock::now();
  uint64_t iterations=0,accepted=0,peak_rejected=0;int epoch=-1;
  auto elapsed=[&](){return std::chrono::duration<double>(Clock::now()-t0).count();};
  std::cout<<"event,seconds,iteration,cost,maxrank,move\n";
  std::cout<<"baseline,0,0,"<<best.cost<<','<<peak(best)<<",none\n";
  save(best,std::string(argv[3])+"/best_order.txt");
  while(elapsed()<budget){
    double seconds=elapsed();int next_epoch=int(seconds/20.0);
    if(next_epoch!=epoch){epoch=next_epoch;cur=best;if(epoch%2){std::reverse(cur.order.begin(),cur.order.end());update(cur);}}
    double phase=std::fmod(seconds,20.0)/20.0;
    double temperature=double(uint64_t(1)<<30)*(.12*std::pow(.004/.12,phase));
    if(epoch>=6)temperature*=.15;
    State cand=cur;int move=rng()%4;int i,j;
    if(rng()%4){i=15+rng()%40;}else i=rng()%N;
    if(move==0){j=std::min(N-1,i+1);if(i==j)--i;}
    else if(rng()%4){j=std::clamp(i+int(rng()%19)-9,0,N-1);}
    else j=rng()%N;
    if(i==j)continue;if(i>j)std::swap(i,j);
    if(move==0||move==3)std::swap(cand.order[i],cand.order[j]);
    else if(move==1){if(rng()%2)std::rotate(cand.order.begin()+i,cand.order.begin()+i+1,cand.order.begin()+j+1);else std::rotate(cand.order.begin()+i,cand.order.begin()+j,cand.order.begin()+j+1);}
    else std::reverse(cand.order.begin()+i,cand.order.begin()+j+1);
    update(cand,i,j);++iterations;
    if(peak(cand)>30){++peak_rejected;continue;}
    double difference=double(cand.cost)-double(cur.cost);
    if(difference<=0||std::generate_canonical<double,53>(rng)<std::exp(-difference/temperature)){cur=cand;++accepted;}
    if(cur.cost<best.cost){best=cur;save(best,std::string(argv[3])+"/best_order.txt");std::cout<<"improvement,"<<elapsed()<<','<<iterations<<','<<best.cost<<','<<peak(best)<<','<<move<<std::endl;}
  }
  std::ofstream result(std::string(argv[3])+"/search_summary.json");
  result<<"{\n\"seconds\":"<<elapsed()<<",\n\"iterations\":"<<iterations<<",\n\"accepted\":"<<accepted<<",\n\"peak_rejected\":"<<peak_rejected<<",\n\"baseline_cost\":"<<start.cost<<",\n\"best_cost\":"<<best.cost<<",\n\"best_maxrank\":"<<peak(best)<<",\n\"seed\":20261030,\n\"budget_seconds\":"<<budget<<"\n}\n";
  std::cout<<"final,"<<elapsed()<<','<<iterations<<','<<best.cost<<','<<peak(best)<<",none\n";
}
