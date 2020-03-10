[TOC]

# 图论

## 最短路

```c++
int n,nxt[maxn],head[maxn],to[maxn],vis[maxn],cnt,m,s;
ll len[maxn],dis[maxn];
void add(int u,int v,ll w){
    nxt[++cnt]=head[u];
    head[u]=cnt;
    len[cnt]=w;
    to[cnt]=v;
}

void dj(int s){
    for(int i=1;i<=n;i++){
        dis[i]=1e18;
    }
    priority_queue<pair<ll,int> >q;
    q.push({0,s});dis[s]=0;
    while(!q.empty()){
        ll t=q.top().first;
        int u=q.top().second;
        q.pop();
        if(vis[u]) continue;
        vis[u]=1;
        for(int i=head[u];~i;i=nxt[i]){
            int v=to[i];
            if(!vis[v] && dis[v]>dis[u]+len[i]){
                dis[v]=dis[u]+len[i];
                q.push({-dis[v],v});
            }
        }
    }
}
```



## 最小生成树

```c++
struct Node{
	int u,v,w;
	friend bool operator <(Node a,Node b){
		return a.w < b.w;
	}
};
Node a[maxn];

int fa[maxn];
int find(int x){
	return fa[x]==x?x:fa[x]=find(fa[x]);
}

int Kruscal(int n,int m){
	for(int i=1;i<=n;i++){
		fa[i] = i;
	}
	sort(a,a+m);
	int ans = 0;
	for(int i=0;i<m;i++){
		int u = a[i].u;
		int v = a[i].v;
		u = find(u); v = find(v);
		if(u!=v){
			fa[u] = v;
			ans += a[i].w;
		}
	}
	return ans;
}
```



## 点分治(链式前向星 + 树的重心)

```c++
struct edge{
    int w,to,next;
}edge[maxn];
int head[maxn],m,n,T;
 
void add_edge(int u,int v,int w){
    edge[++m].next=head[u];
    head[u]=m;
    edge[m].to=v;
    edge[m].w=w;
}
 
int vis[maxn],a[maxn],dis[maxn],len,ans[maxn];
namespace cent{ ///求树的重心的封装函数
    int n,rt,son[maxn],maxl;
    void dfs(int u,int fa){
        son[u]=1;
        int ml=0;
        for(int i=head[u];~i;i=edge[i].next){
            int v=edge[i].to;
            int w=edge[i].w;
            if(v==fa || vis[v]) continue;
            dfs(v,u);
            son[u]+=son[v];
            Smax(ml,son[v]-1);
        }
        Smax(ml,n-son[u]);
        if(ml<maxl){
            maxl=ml;rt=u;
        }
    }
    int GetCent(int x){
        maxl=0x3f3f3f3f;
        dfs(x,-1);
        return rt;
    }
}
 
void getdis(int u,int fa){
    a[++len]=dis[u];
    for(int i=head[u];~i;i=edge[i].next){
        int v=edge[i].to;
        int w=edge[i].w;
        if(vis[v] || v==fa) continue;
        dis[v]=dis[u]+w;
        getdis(v,u);
    }
}
 
int calc(int u,int w,int f){
    dis[u]=w;
    len=0;
    getdis(u,-1);
    for(int i=1;i<len;i++){
        for(int j=i+1;j<=len;j++){
            ans[a[i]+a[j]]+=f;
        }
    }
}
 
void solve(int u){
    calc(u,0,1); //以这个点为中心记录答案
    vis[u]=1;
    for(int i=head[u];~i;i=edge[i].next){
        int v=edge[i].to;
        int w=edge[i].w;
        if(vis[v]) continue;
        calc(v,w,-1); //容斥 删掉部分不符合答案的
        cent::n=cent::son[u];
        int root=cent::GetCent(v);
        solve(root);
    }
}
```



## 倍增求树上最近公共祖先

```c++
struct node{
	int to,next;
}Tree[maxn<<1];
int cnt = 0,head[maxn];

void add(int u,int v){
	Tree[cnt] = node{v,head[u]};
	head[u] = cnt++;
}
int anc[maxn][25],dep[maxn];

void DFS(int u,int fa){
	dep[u] = dep[fa] + 1;
	anc[u][0] = fa;
	for(int i=1;(1<<i)<=dep[u];i++){
		anc[u][i] = anc[anc[u][i-1]][i-1];
	}
	for(int i=head[u];~i;i=Tree[i].next){
		int v = Tree[i].to;
		if( v == fa){
			continue;
		}
		DFS(v,u);
	}
}

int lca(int u,int v){
	if(dep[u] > dep[v]){
		swap(u,v);
	}
	
	for(int i=20;i>=0;i--){
		if((1<<i) <= dep[v] - dep[u]){
			v = anc[v][i];
		}
	}
	
	if(u == v) return u;
	
	for(int i=20;i>=0;i--){
		if(anc[u][i] != anc[v][i]){
			u = anc[u][i];
			v = anc[v][i];
		}
	}
	return anc[u][0];
}
```



# 字符串算法

## KMP模式匹配算法

```c++
int nxt[maxn];
void build_next(char *s){
	int len = strlen(s+1);
	for(int i=2,j=0;i<=len;i++){ // j为之前已匹配成功的长度 
		while(j && s[i] != s[j+1]){
			j = nxt[j];
		}
		if(s[j+1] == s[i]){
			j ++;
		}
		nxt[i] = j;
        if(s[i+1] == s[j+1]) nxt[i] = nxt[j];
	}
}
 
void Kmp(char *T,char *P){
	build_next(P);
	int la = strlen(T+1);
	int lb = strlen(P+1);
	for(int i=1,j=0;i<=la;i++){ // j为之前已匹配成功的长度
		while(j && T[i] != P[j+1]){ // 通过失败指针找到可以成功的位置或找不到 
			j = nxt[j];
		}
		if(T[i] == P[j+1]){ // 匹配成功  匹配成功长度+1 
			j ++;
		}
		if(j == lb){
			printf("%d\n",i-lb+1);
			j = nxt[j];
		}
	}
}
```



## Manacher 最长回文子串算法

```c++
char s[maxn],str[maxn];
int n,len[maxn]; // 以i为中心的最长回文串的半径 
 
void init(){
	str[0] = str[1] = '#'; // 将字符串预处理出来 
	for(int i=0;i<n;i++){
		str[2*i+2] = s[i];
		str[2*i+3] = '#';
	}
	n = n*2 + 2;
	str[n] = 0;
}
 
int Manacher(){
	init();
	int mx=0,id=0,ans=0;
	for(int i=1;i<n;i++){
		if(i<mx){
			len[i] = min(len[2*id-i],mx-i); //该点的半径在对称点的基础上延伸 
		}else{
			len[i] = 1;
		}
		for(;str[i-len[i]] == str[i+len[i]];len[i]++);
		if(i+len[i] > mx){ // 记录之前找到的最长回文子串的延伸的最右边和中心点 
			mx = i+len[i];
			id = i;
		}
		ans = max(ans,len[i]);
	}
	return ans - 1; // 注意 -1 才是真正的回文子串的长度
}
```



## AC自动机(多模式串匹配)

```c++
struct node{
	int son[26],end;
}tire[maxn];
int cnt,fail[maxn];

void ins(char *s,int id){
	int root = 0; // 根节点设为0
	for(int i = 0;s[i];i++){
		int u = s[i] - 'a';
		if(tire[root].son[u] == 0){
			tire[root].son[u] = ++cnt;
		}
		root = tire[root].son[u];
	}
	tire[root].end = id;
}

void build(){
	queue<int>que;
	for(int i=0;i<26;i++){
		if(tire[0].son[i]){
			que.push(tire[0].son[i]);
			fail[tire[0].son[i]] = 0;
		}
	}
	while(!que.empty()){
		int u = que.front();que.pop();
		for(int i=0;i<26;i++){
			if(tire[u].son[i]){ // 找最长后缀
				fail[tire[u].son[i]] = tire[fail[u]].son[i];
				que.push(tire[u].son[i]);
			}else{
				tire[u].son[i] = tire[fail[u]].son[i];
			}
		}
	}
}

int vis[maxn];
int query(char *s){
	int root = 0,ans = 0;
	for(int i=0;s[i];i++){
		root = tire[root].son[s[i]-'a'];
		for(int j = root ;j && ~vis[j]; j = fail[j]){
			ans += tire[j].end;
            vis[j] = -1;
		}
	}
    return ans;
}
```

## 后缀数组

```c++
char s[maxn];
int sa[maxn],rk[maxn],tp[maxn],tax[maxn],Height[maxn];

void Qsort(int n,int m){
	for(int i=0;i<=m;i++) tax[i] = 0;
	for(int i=1;i<=n;i++) ++tax[rk[i]];
	for(int i=1;i<=m;i++) tax[i] += tax[i-1];
	for(int i=n;i>=1;i--) sa[tax[rk[tp[i]]]--] = tp[i];
}

void Suffixsort(int n,int m){
	for(int i=1;i<=n;i++){
		rk[i] = s[i] - '0';
		tp[i] = i;
	}
	Qsort(n,m);
	for(int k=1,p=0;p<n;k<<=1,m=p){
		p = 0;
		for(int i=n;i>=n-k+1;i--) tp[++p] = i;
		for(int i=1;i<=n;i++){
			if(sa[i] > k){
				tp[++p] = sa[i] - k;
			}
		}
		Qsort(n,m);
		swap(tp,rk);
		rk[sa[1]] = p = 1;
		for(int i=2;i<=n;i++){
			rk[sa[i]] = (tp[sa[i]] == tp[sa[i-1]] && tp[sa[i]+k] == tp[sa[i-1]+k])?p:++p;
		} 
	}
	int k = 0;
    for(int i = 1; i <= n; i++) { // get Height rk为i和i-1的最长公共前缀 
        if(k) k--;
        int j = sa[rk[i] - 1];
        while(s[i + k] == s[j + k]) k++;
        Height[rk[i]] = k;
    }
}

int Lcp(int x,int y){ // ST表维护height区间最小值 O(1)求两后缀的最长公共前缀
	return QueryMin(x+1,y);
}
```

## 回文树

```c++
struct Pam{
	int fail[maxn];//失配后跳到的最长回文后缀
	int son[maxn][26];// 两侧添加c到达的状态 
	int cnt[maxn]; //某状态的访问次数 
	int len[maxn];// 某状态的回文串的长度 
	int s[maxn];
    int num[maxn];//fail指针的深度 以某个位置结尾的回文串的数量
	int sz,n;// 回文树大小 插入的字符的数量 
	int last;// last:上一个字符在哪个状态 
	int newNode(int l){
		for(int i=0;i<26;i++) son[sz][i] = 0;
		cnt[sz] = 0;
		len[sz] = l;
		return sz++;
	}
	void init(){
		sz = 0;
		newNode(0);newNode(-1);// 创建偶奇回文树的根 
		last = n = 0;
		s[0] = -1;
		fail[0] = 1;
	}
	int getFail(int x){
		while(s[n-len[x]-1] != s[n]){
			x = fail[x];
		}
		return x;
	}
	void ins(int x){
		s[++n] = x;
		int cur = getFail(last);
		if(son[cur][x] == 0){
			int now = newNode(len[cur] + 2);
			fail[now] = son[getFail(fail[cur])][x];
            num[now] = num[fail[now]] + 1;
			son[cur][x] = now;
		}
		last = son[cur][x];
		cnt[last] ++;
	}
	void Count(){
		for(int i=sz-1;i>=2;i--){
			cnt[fail[i]] += cnt[i];
		}
	}
}Pam;
```

## 后缀自动机

```c++
struct Sam{
	int trans[maxn][26],fail[maxn];
	int maxlen[maxn],minlen[maxn];
	int sz,last,num[maxn];
	Sam(){
		sz = last = 1;
	}
	void ins(int id){
		int cur = ++sz,p = last;
		num[cur] = 1;
		maxlen[cur] = maxlen[last] + 1;
		while(p && trans[p][id] == 0){
			trans[p][id] = cur;
			p = fail[p];
		}
		if(p == 0){
			fail[cur] = 1;
		}else{
			int q = trans[p][id];
			if(maxlen[q] == maxlen[p] + 1){
				fail[cur] = q;
			}else{
				int clone = ++sz;
				memcpy(trans[clone],trans[q],sizeof trans[q]);
				fail[clone] = fail[q];
				fail[cur] = fail[q] = clone;
				maxlen[clone] = maxlen[p] + 1;
				while(p && trans[p][id] == q){
					trans[p][id] = clone;
					p = fail[p];
				}
			}
		}
		last = cur;
	}
	ll calc(){
		ll ans = 0;
		for(int i=2;i<=sz;i++){
			minlen[i] = maxlen[fail[i]] + 1;
			ans += maxlen[i] - minlen[i] + 1;//本质不同子串的个数 
		}
		return ans;
	} 
}Sam;
```



# 计算几何

## 判断两条线段是否相交

```c++
struct Node
{
    double x,y;
}p[maxn];
bool judge(Node a,Node b,Node c,Node d){
    if(!(min(a.x,b.x)<=max(c.x,d.x)&&min(c.x,d.x)<=max(a.x,b.x)&&min(a.y,b.y)<=max(c.y,d.y)&&min(c.y,d.y)<=max(a.y,b.y)))
        return false;
     
    double u,v,w,z;
    u=(c.x-a.x)*(b.y-a.y)-(b.x-a.x)*(c.y-a.y);
    v=(d.x-a.x)*(b.y-a.y)-(b.x-a.x)*(d.y-a.y);
    w=(a.x-c.x)*(d.y-c.y)-(d.x-c.x)*(a.y-c.y);
    z=(b.x-c.x)*(d.y-c.y)-(d.x-c.x)*(b.y-c.y);
 
    if(u*v<=1e-9&&w*z<=1e-9)
        return true;
    return false;
}
```



# 数论

## Miller_Rabin 素数判定

```c++
ll mul(ll x,ll y,ll mod){
    return (x * y - (long long)(x / (long double)mod * y + 1e-3) *mod + mod) % mod;
}
ll qpow(ll a,ll b,ll mod,ll ans = 1){
    for(a %= mod;b;b>>=1){
        if(b&1) ans = mul(ans,a,mod);
        a = mul(a,a,mod);
    }
    return ans;
}
bool Miller_Rabin(ll n,ll u = 0,int t = 0,int s = 10){
    if(n == 2)return true;
    if(n<2||!(n&1))return false;/// <2 || %2==0
    for(t = 0,u = n-1;!(u&1);t++,u>>=1);///n-1=u*2^t
    while(s--){/// s time
        ll a = rand()%(n-1)+1;
        ll x = qpow(a,u,n);///a^u
        for(int i=0;i<t;i++){
            ll y = mul(x,x,n);/// (a^u)^2
            if(y == 1&&x!=1&&x!=n-1)
                return false;
            x = y;
        }
        if(x!=1)return false;/// (a^p-1)%p != 1
    }
    return true;
}
```



## 求1e11以内的素数的个数

```c++
const ll N = 1e11;
const int MX = 1e6;
int vis[MX + 5],cnt,prime[MX + 5];
ll n,lim,lis[2 * MX + 5];
ll tot,le[MX + 5],ge[MX + 5];
ll G[2 * MX + 5];
inline ll &id(ll x){
    return x <= lim ? le[x] : ge[n / x];
}
void Primeall(){
    for(register int i = 2;i <= MX;++i){
        if(!vis[i])
            prime[++cnt] = i;
        for(register int j = 1;j <= cnt && i * prime[j] <= MX;++j)
        {
            vis[i * prime[j]] = 1;
            if(!(i % prime[j])) break;
        }
    }
}
int Sheryang(){
	Primeall();
    scanf("%lld",&n),lim = sqrt(n);
    for(register ll l = 1,r;l <= n;l = r + 1){
        r = n / (n / l);
        lis[id(n / l) = ++tot] = n / l;
        G[tot] = n / l - 1;
    }
    for(register int k = 1;k <= cnt;++k)
    {
        int p = prime[k];
        ll s = (ll)prime[k] * prime[k];
        for(register int i = 1;lis[i] >= s;++i)
            G[i] -= G[id(lis[i] / p)] - (k - 1);
    }
    printf("%lld\n",G[1]);
    return 0;
}
```



## 大整数质因子分解 O(n^{1/3})

```c++
ll Pollard_Rho(ll n, int c){
    ll i = 1, k = 2, x = rand()%(n-1)+1, y = x;
    while(true){
        i++;
        x = (mul(x, x, n) + c)%n;
        ll p = __gcd((y-x+n)%n,n);
        if(p != 1 && p != n) return p;
        if(y == x) return n;
        if(i == k){
            y = x;
            k <<= 1;
        }
    }
}
map<ll,int>vis;
void Find(ll n, int c){
    if(n == 1) return;
    if(Miller_Rabin(n)){
        vis[n] ++;
        return;
    }
    ll p = n, k = c;
    while(p >= n) p = Pollard_Rho(p, c--);
    Find(p , k); Find(n/p , k);
}
```

## 快速傅里叶变换(FFT)

```c++
int limit=1,l,r[maxn];
void fft(complex<double> *a,int f){
	for(int i=0;i<limit;i++){
		if(i<r[i]){
			swap(a[i],a[r[i]]);
		}
	}
	for(int mid=1;mid<limit;mid<<=1){
		complex<double>tmp(cos(pi/mid),f*sin(pi/mid));
		for(int i=0;i<limit;i+=mid*2){
			complex<double>now(1,0);
			for(int j=0;j<mid;j++,now*=tmp){
				complex<double>x=a[i+j],y=now*a[i+j+mid];
				a[i+j]=x+y;a[i+j+mid]=x-y;
			}
		}
	}
}

complex<double>a[maxn],b[maxn];
void init(){
	while(limit<n+m) limit<<=1,l++;  
	for(int i=0;i<limit;i++){
		r[i]=(r[i>>1]>>1)|((i&1)<<(l-1)); 
	}
	fft(a,1);fft(b,1);
	for(int i=0;i<=limit;i++){
		a[i]=a[i]*b[i];
	}
	fft(a,-1);
}
```

## 数论小技巧(逆元 慢速乘 Fibonacci公约数 费马大定理)

```c++
//线性递推求逆元
inv[i] = (p- p/i) * inv[p%i];

//线性递推求阶乘的逆元
infac[n] = qpow(fac[n],p-2);
for(int i=n-1;i>=0;i--){
    infac[i] = (infac[i+1]*(i+1))%p;
}

//O(1)快(man)速乘 防止爆ll
ll mul(ll x,ll y,ll mod){
	ll tmp=(x*y-(ll)((long double)x/mod*y+1.0e-8)*mod);
	return tmp<0?tmp+mod:tmp;
}

//Lucas定理 C(n,m,p) 当p是小素数时
lucas(n,m,p) = C(n%p,m%p,p) * lucas(n/p,m/p,p);

//Fibonaci
gcd(F[n],F[m]) = F[gcd(n,m)]

```

$$
当n > 2时,关于x,y,z的方程  a^{n} + b^{n} = c^{n} \\
当n = 2时 \\
当a为奇数,a = 2x + 1 , c = x^{2} + (x+1)^{2} , b = c-1 \\
当a为偶数,a = 2x + 2 , c = 1 + (x-1)^{2},b = c - 2
$$



## Lucas(p为素数且p很小)

```c++
ll fac[maxn];
ll qpow(ll a,ll b,int p){
	ll ans = 1;a %= p;
	while(b){
		if(b&1) ans = ans*a%p;
		a = a*a%p;
		b >>= 1;
	}
	return ans;
}
ll C(int n,int m,int p){
	if(n < m) return 0;
	return fac[n]*qpow(fac[m],p-2,p)%p*qpow(fac[n-m],p-2,p)%p;
}
ll Lucas(int n,int m,int p){
	if(m == 0) return 1;
	return C(n%p,m%p,p) * Lucas(n/p,m/p,p) % p;
}
```

## 拓展卢卡斯定理

```c++
void exgcd(ll a,ll b,ll &x,ll &y){
    if (!b) return (void)(x=1,y=0);
    exgcd(b,a%b,x,y);
    ll tmp=x;x=y;y=tmp-a/b*y;
}
ll gcd(ll a,ll b){
    if (b==0) return a;
    return gcd(b,a%b); 
}
inline ll INV(ll a,ll p){
    ll x,y;
    exgcd(a,p,x,y);
    return (x+p)%p;
}
inline ll lcm(ll a,ll b){
    return a/gcd(a,b)*b;
}
inline ll mabs(ll x){
    return (x>0?x:-x);
}
inline ll fast_mul(ll a,ll b,ll p){
    ll t=0;a%=p;b%=p;
    while (b){
        if (b&1LL) t=(t+a)%p;
        b>>=1LL;a=(a+a)%p;
    }
    return t;
}
inline ll fast_pow(ll a,ll b,ll p){
    ll t=1;a%=p;
    while (b){
        if (b&1LL) t=(t*a)%p;
        b>>=1LL;a=(a*a)%p;
    }
    return t;
}
inline ll read(){
    ll x=0,f=1;char ch=gc();
    while (!isdigit(ch)) {if (ch=='-') f=-1;ch=gc();}
    while (isdigit(ch)) x=x*10+ch-'0',ch=gc();
    return x*f;
}
inline ll F(ll n,ll P,ll PK){
    if (n==0) return 1;
    ll rou=1;//循环节
    ll rem=1;//余项 
    for (ll i=1;i<=PK;i++){
        if (i%P) rou=rou*i%PK;
    }
    rou=fast_pow(rou,n/PK,PK);
    for (ll i=PK*(n/PK);i<=n;i++){
        if (i%P) rem=rem*(i%PK)%PK;
    }
    return F(n/P,P,PK)*rou%PK*rem%PK;
}
inline ll G(ll n,ll P){
    if (n<P) return 0;
    return G(n/P,P)+(n/P);
}
inline ll C_PK(ll n,ll m,ll P,ll PK){
    ll fz=F(n,P,PK),fm1=INV(F(m,P,PK),PK),fm2=INV(F(n-m,P,PK),PK);
    ll mi=fast_pow(P,G(n,P)-G(m,P)-G(n-m,P),PK);
    return fz*fm1%PK*fm2%PK*mi%PK;
}
ll A[1001],B[1001];
//x=B(mod A)
inline ll exLucas(ll n,ll m,ll P){ // 求C(n,m)%p p为任意数
    ll ljc=P,tot=0;
    for (ll tmp=2;tmp*tmp<=P;tmp++){
        if (!(ljc%tmp)){
            ll PK=1;
            while (!(ljc%tmp)){
                PK*=tmp;ljc/=tmp;
            }
            A[++tot]=PK;B[tot]=C_PK(n,m,tmp,PK);
        }
    }
    if (ljc!=1){
        A[++tot]=ljc;B[tot]=C_PK(n,m,ljc,ljc);
    }
    ll ans=0;
    for (ll i=1;i<=tot;i++){
        ll M=P/A[i],T=INV(M,A[i]);
        ans=(ans+B[i]*M%P*T%P)%P;
    }
    return ans;
}
```



## 二次同余方程求解(P为奇素数)

$$
x^2 \equiv n (mod \;p)
$$

```c++
ll quick_mod(ll a, ll b, ll m){
    ll ans = 1;
    a %= m;
    while(b){
        if(b & 1){
            ans = ans * a % m;
            b--;
        }
        b >>= 1;
        a = a * a % m;
    }
    return ans;
}
 
struct T{
    ll p, d;
};
 
ll w;
T multi_er(T a, T b, ll m){
    T ans;
    ans.p = (a.p * b.p % m + a.d * b.d % m * w % m) % m;
    ans.d = (a.p * b.d % m + a.d * b.p % m) % m;
    return ans;
}
 
T power(T a, ll b, ll m){
    T ans;
    ans.p = 1;
    ans.d = 0;
    while(b){
        if(b & 1){
            ans = multi_er(ans, a, m);
            b--;
        }
        b >>= 1;
        a = multi_er(a, a, m);
    }
    return ans;
}

ll Legendre(ll a, ll p){
    return quick_mod(a, (p-1)>>1, p);
}
 
ll Mod(ll a, ll m){
    a %= m;
    if(a < 0) a += m;
    return a;
}
 
ll Solve(ll n,ll p){
    if(p == 2) return 1;
    if (Legendre(n, p) + 1 == p)
        return -1;
    ll a = -1, t;
    while(true){
        a = rand() % p;
        t = a * a - n;
        w = Mod(t, p);
        if(Legendre(w, p) + 1 == p) break;
    }
    T tmp;
    tmp.p = a;
    tmp.d = 1;
    T ans = power(tmp, (p + 1)>>1, p);
    return ans.p;
}
 
int main(){
	srand(time(0));
    int t;
    scanf("%d", &t);
    while(t--){
        int n, p;
        scanf("%d %d",&n,&p);
        n %= p;
        int a = Solve(n, p);
        if(a == -1)
        {
            puts("No root");
            continue;
        }
        int b = p - a;
        if(a > b) swap(a, b);
        
        printf("%d %d\n",a,b);
    }
    return 0;
}
```

## 中国剩余定理(非互质)

```c++
int n;
ll a[maxn],b[maxn],x,y;
ll exgcd(ll a, ll b, ll &x, ll &y){
    if(!b){ x = 1; y = 0; return a; }
    ll d = exgcd(b, a % b, y, x);
    y -= (a/b) * x;
    return d;
}

ll Slow_Mul(ll x,ll y,ll mod){
    return (x * y - (long long)(x / (long double)mod * y + 1e-3) *mod + mod) % mod;
}

ll China(){  /// x = bi(mod ai)
	ll ans = a[1];
    ll M = b[1];
    for(int i = 2; i <= n; ++i){
       ll B = (a[i] - ans % b[i] + b[i]) % b[i];
       ll GCD = exgcd(M, b[i], x, y);
       if(B%GCD){
            return -1;
       }
       x = Slow_Mul(x, B / GCD, b[i]/GCD);
       ans += M * x;
       M *= b[i] / GCD;
       ans = (ans%M + M) % M;
    }
    return ans;
}
```

## 拉格朗日插值求解多项式

$$
f(k) = \sum_{i=1}^{n} y[i] * \prod_{i!=j} \frac{k-x[i]}{x[i]-x[j]}
$$

```c++
ll large(int n,int k){
	ll ans = 0;
	for(int i=1;i<=n;i++){
		ll s1 = 1,s2 = 1;
		for(int j=1;j<=n;j++){
			if(i != j){
				s1 = s1 * (k-x[j]+mod) % mod;
				s2 = s2 * (x[i]-x[j]+mod) %mod;
			}
		}
		ans = (ans + y[i]*s1%mod * qpow(s2,mod-2)%mod)%mod;
	}
	return ans;
}
```

## Polya染色定理

```c++
//n个点 n条边的环 涂m种颜色 有多少本质不同的方案数(可旋转)
ll qpow(ll x,ll y){ //快速幂
    ll res = 1;
    while(y){
        if(y&1) res = res*x%mod;
        x = x*x%mod;
        y >>= 1;
    }
    return res;
}

ll euler_phi(ll n) { //欧拉函数
    ll res=1;
    for(ll i=2; i*i<=n; i++)
        if(n%i==0) {
            n/=i,res=res*(i-1);
            while(n%i==0) n/=i,res=res*i;
        }
    if(n>1) res=res*(n-1);
    return res;
}

ll polya(ll n,ll m) { //polya定理主体
    ll tot=0;
    for(ll i=1; i*i<=n; i++) {
        if(n%i) continue;
        tot += euler_phi(i)*qpow(m,n/i-1);
        if(i*i!=n) tot += euler_phi(n/i)*qpow(m,i-1);
    }
    return tot%mod;
}
```

## 杜教筛

```c++
int v[maxn+10],p[maxn+10];
ll phi[maxn+10],mu[maxn+10];
inline void init() {
    v[1]=mu[1]=phi[1]=1;
    int cnt=0;
    for(int i=2;i<=maxn;++i) {
        if(!v[i]) p[++cnt]=i,mu[i]=-1,phi[i]=i-1;
        for(int j=1;j<=cnt&&i*p[j]<=maxn;++j){
            v[i*p[j]]=1;
            if (i%p[j]) mu[i*p[j]]=-mu[i],phi[i*p[j]]=phi[i]*phi[p[j]];
            else{ mu[i*p[j]]=0,phi[i*p[j]]=phi[i]*p[j]; break; }
        }
    }
    for (int i=1;i<=maxn;++i) mu[i]+=mu[i-1],phi[i]+=phi[i-1];
}

unordered_map<int,ll> ansmu,ansphi;
inline ll Getphi(int n){ // 欧拉函数
	if(n <= maxn) return phi[n];
	if(ansphi[n]) return ansphi[n];
	ll ans = 1LL*n*(n+1)/2;
	for(unsigned int l=2,r;l<=n;l=r+1){
		r = n/(n/l);
		ans -= (r-l+1)*Getphi(n/l);
	}
	return ansphi[n] = ans;
}

inline ll Getmu(int n){ // 莫比乌斯函数
	if(n<=maxn) return mu[n];
	if(ansmu[n]) return ansmu[n];
	ll ans = 1;
	for(unsigned int l=2,r;l<=n;l=r+1){
		r = n/(n/l);
		ans -= (r-l+1)*Getmu(n/l);
	}
	return ansmu[n] = ans;
}
```

## 等比数列求和(a + a^2 + a^3 + ... + a^b)

```c++
ll solve(ll a,ll b){
	if(b == 1) return a;
	if(b == 0) return 0;
	ll tmp = solve(a,b/2);
	ll ans = tmp * (1 + qpow(a,b/2)) % mod;
	if(b&1){
		ans = ans + qpow(a,b);
		ans %= mod;
	}
	return ans;
}
```

# 数据结构

## tire树(字典树)

```c++
struct node{
    int son[26],sz,end;
}tire[maxn];
 
int tot; //注意根节点编号为0
void ins(char *s,int id){
    int root = 0;
    tire[root].sz ++;
    for(int i=0;s[i];i++){
        int u = s[i] - 'a';
        if(tire[root].son[u] == 0){
            tire[root].son[u] = ++tot;
        }
        root = tire[root].son[u];
        tire[root].sz ++;
    }
    tire[root].end ++;
}
```



## 可持久化01字典树(求区间异或最大值)

```c++
int cnt,tire[maxn*40][2],sz[maxn*40],root[maxn];
void ins(int pre,int &rt,int x){
	rt = ++cnt;
	int now = rt;
	for(int i=30;i>=0;i--){
		int t = (x>>i)&1;
		tire[now][t] = ++cnt;
		tire[now][t^1] = tire[pre][t^1];
		now = tire[now][t];
		pre = tire[pre][t];
		sz[now] = sz[pre] + 1;
	}
}
int query(int l,int r,int x){
	int ans = 0;
	for(int i=30;i>=0;i--){
		int t = (x>>i)&1;
		if(sz[tire[r][t^1]] - sz[tire[l][t^1]] > 0){
			ans |= (1<<i);
			l = tire[l][t^1];
			r = tire[r][t^1];
		}else{
			l = tire[l][t];
			r = tire[r][t];
		}
	}
	return ans;
}
```



## 主席树(静态区间第K小)

```c++
struct Node{
	int l,r,sum;
}Tree[maxn*30];
int cnt,root[maxn];

void update(int pre,int &rt,int vol,int l,int r){
    rt = ++cnt;
	Tree[rt] = Tree[pre];
	Tree[rt].sum ++;
	if(l==r) return;
	int mid = l+r>>1;
	if(vol<=mid){
		update(Tree[pre].l,Tree[rt].l,vol,l,mid);
	}else{
		update(Tree[pre].r,Tree[rt].r,vol,mid+1,r);
	}
}

int query(int pre,int cur,int k,int l,int r){ //查找区间第k小
	if(l==r){
		return l;
	}
	int sum = Tree[Tree[cur].l].sum - Tree[Tree[pre].l].sum;
	
	int mid = l+r>>1;
	if(sum>=k){
		return query(Tree[pre].l,Tree[cur].l,k,l,mid);
	}else{
		return query(Tree[pre].r,Tree[cur].r,k-sum,mid+1,r);
	}
}
```



## 树链剖分

```c++
int nxt[maxn],head[maxn],to[maxn],cnt;
ll len[maxn];
void add(int u,int v,ll w){
    nxt[++cnt]=head[u];
    head[u]=cnt;
    len[cnt]=w;
    to[cnt]=v;
}

int _dfs[maxn],son[maxn],far[maxn],siz[maxn],sum[maxn];
int dep[maxn],tot,top[maxn],n,a[maxn],id[maxn],tree[maxn];
 
void dfs1(int u,int fa,int depth)
{
    far[u]=fa;
    siz[u]=1;
    dep[u]=depth;
    for(int i=head[u];~i;i = nxt[i]){
        int v = to[i];
        if(v==fa) continue;
        dfs1(v,u,depth+1);
        siz[u]+=siz[v];
        if(siz[v]>siz[son[u]]) son[u]=v; //找到每个节点最大的孩子节点
    }
}
 
void dfs2(int x,int t)
{
    _dfs[x]=++tot; //每个节点的在新的序列的位置
    top[x]=t;
    id[tot]=x; //新的序列
    if(son[x]) dfs2(son[x],t);
    for(int i=head[x];~i;i=nxt[i]){
        int v = to[i];
        if(v!=far[x] && v!=son[x])
            dfs2(v,v);
    }
}

ll cal(int u,int v)  //查询 u->v 这条链上的信息
{
    ll ans=0;
    while(top[u]!=top[v]){
        if(dep[top[u]]<dep[top[v]])
            swap(u,v);
        ans+=query(_dfs[top[u]],_dfs[u]); //用某个数据结构维护一段区间
        u=far[top[u]];
    }
    if(dep[u]>dep[v]){
    	swap(u,v);
	}
    ans += query(_dfs[u],_dfs[v]);
    return ans;
}
```



## 静态区间线性基(在线)

```c++
int bit[maxn][32],pos[maxn][32];
inline void get(int x, int k, int r) {
    
    for(int i=30;i>=0;i--){
        bit[r][i] = bit[r-1][i];
        pos[r][i] = pos[r-1][i];
    }
    
    for (int i = 30; i >= 0; i--)
        if ((x >> i) & 1) {
            if (!bit[r][i]) {
                bit[r][i] = x;
                pos[r][i] = k;
                return;
            }
            if (pos[r][i] < k) {
                swap(k,pos[r][i]);
                swap(x,bit[r][i]);
            }
            x ^= bit[r][i];
        }
}
int query(int l,int r){
    int ans = 0;
    for(int i=30;i>=0;i--){
        if(pos[r][i] >= l){
            ans = max(ans,ans ^ bit[r][i]);
        }
    }
    return ans;
}
```



## ST表(静态RMQ)

```c++
int st[maxn][35],lg[maxn]; 
int n;

void init(){     //初始化很重要
    lg[0] = -1;
    for(int i=1;i<=n;i++){
    	lg[i]=lg[i>>1]+1;
	}
	for(int j=1;(1<<j)<=n;j++){
    	for(int i=1;i+(1<<j)-1<=n;i++){
    		st[i][j]=max(st[i][j-1],st[i+(1<<j-1)][j-1]); //维护最大值的st表
		}
	}
}

int Query(int l,int r){ // 查询l r的最大值
	int d = lg[r-l+1];
    r = r - (1<<d) + 1;
    int ans=max(st[l][d],st[r][d]);
    return ans;
}
```



## 二维ST表

```c++
int a[maxn][maxn],Max[maxn][maxn][10][10],Min[maxn][maxn][10][10];
int Log[maxn] = {-1};
void init(){ // n*n的矩阵
    for (int i = 0; (1 << i) <= n; i++){
        for (int j = 0; (1 << j) <= n; j++){
            if (i == 0 && j == 0) continue;
            for (int row = 1; row + (1 << i) - 1 <= n; row++)
                for (int col = 1; col + (1 << j) - 1 <= n; col++){
                    if (i == 0)
                        Max[row][col][i][j] = max(Max[row][col][i][j - 1], Max[row][col + (1 << (j - 1))][i][j - 1]);
                    else if (j == 0)
                        Max[row][col][i][j] = max(Max[row][col][i - 1][j], Max[row + (1 << (i - 1))][col][i - 1][j]);
                    else
                        Max[row][col][i][j] = max(Max[row][col][i][j - 1], Max[row][col + (1 << (j - 1))][i][j - 1]);
                }
        }
    }
    for (int i = 0; (1 << i) <= n; i++){
        for (int j = 0; (1 << j) <= n; j++){
            if (i == 0 && j == 0) continue;
            for (int row = 1; row + (1 << i) - 1 <= n; row++)
                for (int col = 1; col + (1 << j) - 1 <= n; col++){
                    if (i == 0)
                        Min[row][col][i][j] = min(Min[row][col][i][j - 1], Min[row][col + (1 << (j - 1))][i][j - 1]);
                    else if (j == 0)
                        Min[row][col][i][j] = min(Min[row][col][i - 1][j], Min[row + (1 << (i - 1))][col][i - 1][j]);
                    else
                        Min[row][col][i][j] = min(Min[row][col][i][j - 1], Min[row][col + (1 << (j - 1))][i][j - 1]);
                }
        }
    }
}
int Query(int x1, int y1, int x2, int y2){
    int kx = 0, ky = 0;
    kx = Log[x2-x1+1];
    ky = Log[y2-y1+1];
    int m1 = Max[x1][y1][kx][ky];
    int m2 = Max[x2 - (1 << kx) + 1][y1][kx][ky];
    int m3 = Max[x1][y2 - (1 << ky) + 1][kx][ky];
    int m4 = Max[x2 - (1 << kx) + 1][y2 - (1 << ky) + 1][kx][ky];
 
    int m5 = Min[x1][y1][kx][ky];
    int m6 = Min[x2 - (1 << kx) + 1][y1][kx][ky];
    int m7 = Min[x1][y2 - (1 << ky) + 1][kx][ky];
    int m8 = Min[x2 - (1 << kx) + 1][y2 - (1 << ky) + 1][kx][ky];
    return max(max(m1, m2), max(m3, m4));
	return min(min(m5, m6), min(m7, m8));
}
// 预处理log Log[i] = Log[i>>1] + 1;
```



## 笛卡尔树(维护不相交区间最值)

```c++
int top,st[maxn],ls[maxn],rs[maxn];
void init(int x){ //小顶堆
	while(top && a[x] < a[st[top]]){ //维护右链
		ls[x] = st[top] , top --;
	}
	if(top) rs[st[top]] = x;
	st[++top] = x;
}
// while(top) root = st[top--];
```

## 线段树扫描线

```c++
struct Node{
	int l,r;
	int cnt,len;
}T[maxn];

int n,lisan[maxn];
void pushup(int rt){
	if(T[rt].cnt){
		T[rt].len = lisan[T[rt].r+1] - lisan[T[rt].l];
	}else{
		T[rt].len = T[rt<<1].len + T[rt<<1|1].len;
	}
}

void update(int L,int R,int vol,int rt=1){
	if(T[rt].l >= L && T[rt].r <= R){
		T[rt].cnt += vol;
		pushup(rt);
		return;
	}
	int mid = T[rt].l + T[rt].r >>1;
	if(L <= mid){
		update(L,R,vol,rt<<1);
	}
	if(R > mid){
		update(L,R,vol,rt<<1|1);
	}
	pushup(rt);
}

void build(int l,int r,int rt){
	T[rt].l = l,T[rt].r = r;
	if(l == r) return;
	int mid = l+r>>1;
	build(l,mid,rt<<1);
	build(mid+1,r,rt<<1|1);
}

struct Mat{
	int x,low,high,flag;
	friend bool operator <(Mat a,Mat b){
		return a.x < b.x;
	}
}M[maxn];

int Sheryang(){

	n = read;
	int cnt = 0,cnt_ = 0;
	for(int i=1;i<=n;i++){
		int x_ = read;
		int y_ = read;
		int x__ = read;
		int y__ = read;
		M[++cnt].x = x_;
		M[cnt].low = y_;
		M[cnt].high = y__;
		M[cnt].flag = 1;

		M[++cnt].x = x__;
		M[cnt].low = y_;
		M[cnt].high = y__;
		M[cnt].flag = -1;

		lisan[++cnt_] = y_;
		lisan[++cnt_] = y__;
	}
	sort(lisan+1,lisan+1+cnt_);
	cnt_ = unique(lisan+1,lisan+1+cnt_) - lisan - 1;
	for(int i=1;i<=n<<1;i++){
		int pos1 = lower_bound(lisan+1,lisan+1+cnt_,M[i].low) - lisan;
		int pos2 = lower_bound(lisan+1,lisan+1+cnt_,M[i].high) - lisan;
		M[i].low = pos1;M[i].high = pos2;
	}

	sort(M+1,M+1+2*n);
	build(1,n<<1,1);
	ll ans = 0;
	for(int i=1;i<n<<1;i++){ // 线段树节点x对应 [T[x].l,T[x].r+1] 这条横边
		update(M[i].low,M[i].high-1,M[i].flag);
		ans += 1LL * T[1].len * (M[i+1].x - M[i].x); 
	}
	printf("%lld\n",ans);
    return 0;
}
```



# 博弈

## SG函数打表

```c++
//f[]：可以取走的石子个数
//sg[]:0~n的SG函数值
//Hash[]:mex{}
int f[maxn],sg[maxn],Hash[maxn];
void initSG(int n){
    memset(sg,0,sizeof(sg));
    for(int i=1;i<=n;i++){//当前堆中有i个石头
    
        memset(Hash,0,sizeof(Hash));
        for(int j=1;f[j]<=i;j++){
			Hash[sg[i-f[j]]]=1;
		}
		
        for(int j=0;j<=n;j++){
			if(Hash[j]==0){//求mes{}中未出现的最小的非负整数
            	sg[i]=j;
           		break;
        	}
        }
    }
}
```

## 威佐夫博弈

```c++
// 两堆石子 一堆为n 一堆为m 可以从任意一堆中取任意数量 也可以从两堆中拿相同数量 先取完则获胜
// 当 n/(n-m) == (sqrt(5)+1)/2 时必败 否则必胜
```

# 动态规划

## 数位dp

```c++
// 求小于等于n的数字且能被所有位上的数之和整除的数的个数
// 数位dp
ll dp[20][105][150];
int a[20],mod;
ll dfs(int len,int sum,int remain,bool limit){//目前枚举的位数，各位和，取余mod的余数，限制
    if(len==0) return sum==mod && remain==0; //当各位和与自己期望的相同且枚举的数字是各位和的倍数就返回1
    if(!limit && dp[len][sum][remain]!=-1) return dp[len][sum][remain];//记忆化记录的是没有限制的普遍情况，如果有限制，记忆化的一定是错的
    int up=limit?a[len]:9;//有限制则上限是a[len],没有限制是9
    ll cur=0;
    for(int i=0;i<=up;i++){
        if(i+sum>mod) break; //剪枝，和已经大于期望的mod，则直接结束
        cur+=dfs(len-1,sum+i,(remain*10+i)%mod,limit && i==a[len]);//当这位有限制且已枚举到上限则下一位有限制
    }
    return limit?cur:dp[len][sum][remain]=cur;
}
 
ll solve(ll x){
    memset(a,0,sizeof(a));
    int k=0;
    while(x){
        a[++k]=x%10;
        x/=10;
    }
    ll ans=0;
    for(int i=1;i<=9*k;i++){//枚举期望的和
        memset(dp,-1,sizeof(dp));//每次枚举的只是这次的期望和
        mod = i;
        ans+=dfs(k,0,0,true);
    }
    return ans;
}
 
```

# 矩阵

## 高斯消元

```c++
const double eps = 1e-7;
double mat[105][105],x[105];
int Guess(int n){
	
	for(int i=1;i<=n;i++){
		int pos = i;
		for(int j=i;j<=n;j++){
			if(mat[i][j] > mat[i][pos]){
				pos = j;
			}
		}
		if(fabs(mat[i][pos]) < eps){
			printf("No Solution\n");
			return 0;
		}
		if(pos != i){
			swap(mat[pos],mat[i]);
		}
		double div = mat[i][i];
		for(int j=i;j<=n+1;j++){
			mat[i][j] /= div;
		}
		for(int j=i+1;j<=n;j++){
			div = mat[j][i];
			for(int k=i;k<=n+1;k++){
				mat[j][k] -= mat[i][k]*div;
			}
		}
	}
	
	x[n] = mat[n][n+1];
	for(int i=n-1;i>=1;i--){
		x[i] = mat[i][n+1];
		for(int j=i+1;j<=n;j++){
			x[i] -= mat[i][j]*x[j];
		}
	}
	return 1;
}
```



## 最大的全1子矩阵

```c++
ll calc(){
     
    for(int i=1;i<=n;i++){
        for(int j=1;j<=m;j++){
            num[i][j] = (a[i][j])?num[i][j-1]+1:0;
        }
    }
      
    int top = 0;
    ll Max = 0; 
    for(int j=1;j<=m;j++){
        top = 0;
        for(int i=1;i<=n;i++){
            if(num[i][j]){
                up[i] = 1;
                while(top && num[i][j]<num[st[top]][j]) up[i]+=up[st[top--]]; 
                st[++top] = i;
            }else{
                top = 0;
                up[i] = 0;
            }
        }
        
        top = 0;
        for(int i=n;i>=1;i--){
            if(num[i][j]){
                down[i] = 1;
                while(top && num[i][j]<=num[st[top]][j]) down[i]+=down[st[top--]]; 
                st[++top] = i;
            }else{
                top = 0;
                down[i] = 0;
            }
            
            ll tmp = (up[i] + down[i] - 1)*num[i][j];
			Max = max(Max,tmp);
        }
    }
    return Max;
}
```



## 矩阵快速幂

```c++
struct mat{
    ll m[6][6];
    inline ll* operator [] (int pos) {
	    return m[pos];
	}
	inline void clr(){
	    memset(m, 0, sizeof m);
	}
};
mat operator *(mat a,mat b){
    mat ans;
    for(int i=1;i<=5;i++){
        for(int j=1;j<=5;j++){
            ans[i][j] = 0;
            for(int k=1;k<=5;k++){
                ans[i][j] = (ans[i][j] + a[i][k]*b[k][j]) % mod;
            }
        }
    }
    return ans;
}

mat qpow(mat a,ll b){
	mat ans;ans.clr();
    for(int i=1;i<=5;i++){
        ans[i][i] = 1;
    }
    while(b > 0){
        if(b&1) ans=ans*a;
        a=a*a;b>>=1;
    }
    return ans;
}
```



## 杜教BM

```c++
#include<bits/stdc++.h>
using namespace std;
#define rep(i,a,n) for (int i=a;i<n;i++)
#define pb push_back
typedef long long ll;
#define SZ(x) ((ll)(x).size())
typedef vector<ll> VI;
typedef pair<ll,ll> PII;
const ll mod=1000000007;
ll powmod(ll a,ll b) {ll res=1;a%=mod; assert(b>=0); for(;b;b>>=1){if(b&1)res=res*a%mod;a=a*a%mod;}return res;}
// head
 
ll _,n;
namespace linear_seq {
    const ll N=10010;
    ll res[N],base[N],_c[N],_md[N];
 
    vector<ll> Md;
    void mul(ll *a,ll *b,ll k) {
        rep(i,0,k+k) _c[i]=0;
        rep(i,0,k) if (a[i]) rep(j,0,k) _c[i+j]=(_c[i+j]+a[i]*b[j])%mod;
        for (ll i=k+k-1;i>=k;i--) if (_c[i])
            rep(j,0,SZ(Md)) _c[i-k+Md[j]]=(_c[i-k+Md[j]]-_c[i]*_md[Md[j]])%mod;
        rep(i,0,k) a[i]=_c[i];
    }
    ll solve(ll n,VI a,VI b) { 

        ll ans=0,pnt=0;
        ll k=SZ(a);
        assert(SZ(a)==SZ(b));
        rep(i,0,k) _md[k-1-i]=-a[i];_md[k]=1;
        Md.clear();
        rep(i,0,k) if (_md[i]!=0) Md.push_back(i);
        rep(i,0,k) res[i]=base[i]=0;
        res[0]=1;
        while ((1ll<<pnt)<=n) pnt++;
        for (ll p=pnt;p>=0;p--) {
            mul(res,res,k);
            if ((n>>p)&1) {
                for (ll i=k-1;i>=0;i--) res[i+1]=res[i];res[0]=0;
                rep(j,0,SZ(Md)) res[Md[j]]=(res[Md[j]]-res[k]*_md[Md[j]])%mod;
            }
        }
        rep(i,0,k) ans=(ans+res[i]*b[i])%mod;
        if (ans<0) ans+=mod;
        return ans;
    }
    VI BM(VI s) {
        VI C(1,1),B(1,1);
        ll L=0,m=1,b=1;
        rep(n,0,SZ(s)) {
            ll d=0;
            rep(i,0,L+1) d=(d+(ll)C[i]*s[n-i])%mod;
            if (d==0) ++m;
            else if (2*L<=n) {
                VI T=C;
                ll c=mod-d*powmod(b,mod-2)%mod;
                while (SZ(C)<SZ(B)+m) C.pb(0);
                rep(i,0,SZ(B)) C[i+m]=(C[i+m]+c*B[i])%mod;
                L=n+1-L; B=T; b=d; m=1;
            } else {
                ll c=mod-d*powmod(b,mod-2)%mod;
                while (SZ(C)<SZ(B)+m) C.pb(0);
                rep(i,0,SZ(B)) C[i+m]=(C[i+m]+c*B[i])%mod;
                ++m;
            }
        }
        return C;
    }
    ll gao(VI a,ll n) {
        VI c=BM(a);
        c.erase(c.begin());
        rep(i,0,SZ(c)) c[i]=(mod-c[i])%mod;
        return solve(n,c,VI(a.begin(),a.begin()+SZ(c)));
    }
};
 
int main() {
    int T;
    scanf("%d",&T);
    while (T--)
    {
        scanf("%lld",&n);
        vector<ll>v;
        v.push_back(3);
        v.push_back(9);
        v.push_back(20);
        v.push_back(46);
        v.push_back(106);
        v.push_back(244);
        v.push_back(560);
        v.push_back(1286);
        v.push_back(2956);
        v.push_back(6794);
         
        printf("%lld\n",linear_seq::gao(v,n-1));
    }
}
```



# 构造

## Hilbert曲线

```c++
void rot(int n, int *x, int *y, int rx, int ry);

//XY坐标到Hilbert代码转换
int xy2d (int n, int x, int y){
    int rx, ry, s, d=0;
    for (s=n/2; s>0; s/=2){
        rx = (x & s) > 0;
        ry = (y & s) > 0;
        d += s * s * ((3 * rx) ^ ry);
        rot(s, &x, &y, rx, ry);
    }
    return d;
}

//Hilbert代码到XY坐标
void d2xy(int n, int d, int *x, int *y){
    int rx, ry, s, t=d;
    *x = *y = 0;
    for (s=1; s<n; s*=2){
        rx = 1 & (t/2);
        ry = 1 & (t ^ rx);
        rot(s, x, y, rx, ry);
        *x += s * rx;
        *y += s * ry;
        t /= 4;
    }
}

void rot(int n, int *x, int *y, int rx, int ry){
    if (ry == 0){
        if (rx == 1){
            *x = n-1 - *x;
            *y = n-1 - *y;
        }

        //Swap x and y
        int t  = *x;
        *x = *y;
        *y = t;
    }
}
```



## 对拍函数

```c++
#include<bits/stdc++.h>
using namespace std;
int main(){
	while(1){
		system("input.exe");
		system("Ac.exe");
		system("Wa.exe");
		if(system("fc out1.txt out2.txt"))break;
	}
	return 0;
}
```

## 回旋矩阵求值

```c++
ll getval(int x,int y, int n){ // 正常的 x,y坐标系
	ll t = min(min(x, y), min(n - x + 1, n - y + 1));
	ll ta = 4 * (t - 1) * (n - t + 1);
	if (x == n - t + 1) ta += n - t - y + 2;
	else if (x == t) ta += 2 * n - 5 * t + y + 3;
	else if (y == t) ta += 2 * n - 3 * t - x + 3;
	else ta += 3 * n - 7 * t + x + 4;
	return ta;
}
// 三阶回旋矩阵示例
//  7 8 1
//  6 9 2
//  5 4 3
```

## C++加速 IO

```c++
ios::sync_with_stdio(false);
cin.tie(0);
```

