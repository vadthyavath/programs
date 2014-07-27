#include <iostream>
#include <string>
using namespace std;
//program to Generate Binary Numbers from 1 to n
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