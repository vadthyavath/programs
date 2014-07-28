#include <iostream>
#include <string>
using namespace std;
//http_proxy=http://your.proxy.server:proxy_port
//git config --global http.proxy $http_proxy
//git clone http://git.gnome.org/browse/tracker


//program to Generate Binary Numbers from 1 to n
//g++ 4.8.1 doesn't support to_string method so we use old version of g++ in the following way
//g++ -std=c++11 filename.cpp -o outputfilename
int main()
{	int a;
	cout<<"enter number";

	cin>>a;
	if (a>=1){cout<<1<<"\n";}
		for (int i=2;i<=a;i++)
				{	int m=i;
					string str1="";
					while(true)
						{ 
							str1=to_string(m%2)+str1;
							m=m/2;
							if(m==1)
							 {str1=to_string(1)+str1;
								break;
							 }
						}
					cout<<str1<<"\n";
				}


}