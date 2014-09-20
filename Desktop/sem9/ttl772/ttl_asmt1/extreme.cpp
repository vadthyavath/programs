#include <iostream>
#include <stdlib.h>
#include <limits>
using namespace std;
int main()
{
   while (1)
   {  try { 
   int n,mini=10000000,max=-10000000,k=0;
   cout<<"enter sequence\n";
   while (1)
   {
   cin>>n;
   if(!cin){++k;
         cin.clear();
         cin.ignore (std::numeric_limits<std::streamsize>::max(), '\n');
         throw 20;
         break;

      }
   if (n!=0)
   {if (n>max)   
         max=n;
   
   if (n<mini)
      mini=n;

   }
   else
   {cin.clear();
         cin.ignore (std::numeric_limits<std::streamsize>::max(), '\n');
      
      break;
   }
 
}
if (k==0 && ((mini!=10000000) || (max!=-10000000)))
 cout<<"extreme values are\t"<<mini<<"\t"<<max<<"\n";

}
catch (int c)
{  //cin.clear();  
   //cin.ignore (std::numeric_limits<std::streamsize>::max(), '\n');
   cout<<"input error";
   //cin.clear(e);
}


int stop;
cout<<"\npress 9 to stop/any no. to continue\n";
cin>>stop;
if (stop==9)
   break;


}
 
   return 0;
}