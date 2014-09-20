#include <iostream>
#include <limits>
using namespace std;
int main()
{while(1)
  {
  try {
 int del,n,i,j,repeat=0;
 cout<<"enter array size";
 cin>>n;
 if(!cin) {
      cin.clear();
      cin.ignore (std::numeric_limits<std::streamsize>::max(), '\n');
      throw 20;
      }
 int a[n];
 cout<<"enter elements\n";
 for(i=0;i<n;i++){
  cin>>a[i];
if(!cin) {
      cin.clear();
      cin.ignore (std::numeric_limits<std::streamsize>::max(), '\n');
      throw 20;
      }}
 cout<<"enter the no. to delete";
 cin>>del;
 if(!cin) {
      cin.clear();
      cin.ignore (std::numeric_limits<std::streamsize>::max(), '\n');
      throw 20;
      }
 for (int k=0;k<n;k++)
 {
  if(a[i]=del)
   repeat++;
 }
 if (repeat!=0){
int b[n-repeat];
  for(i=0,j=0;i<n;i++)
 {
  if(a[i]!=del)
   b[j++]=a[i];
 }

  if(j==n)
 {
  cout<<"element not in array";
 }
 else
 {
  cout<<"\nNew Array is ";
  for(i=0;i<j;i++)
   cout<<b[i]<<" ";
 }
}
}
catch (int c)
{

  cout<<"input error";
}

int stop;
cout<<"\npress 9 to stop/any no. to continue\n";
cin>>stop;
if (stop==9)
     break;
   else continue;
 return 0;
}
}