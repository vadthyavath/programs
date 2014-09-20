#include <iostream>
#include <stdlib.h>
#include <limits>
using namespace std;

int main()
{
   while (1)
   {  try {
   int n,i,j,k,temp, current_state, swap;
   cout<<"Enter the array size\n";
   cin>>n;
   if(!cin) {
      cin.clear();
             cin.ignore (std::numeric_limits<std::streamsize>::max(), '\n');

        throw 20;
      }
   if (n>0){
   int arr[n]; 
   cout<<"enter the elements to be sorted\n";
 
   for ( i = 0 ; i < n ; i++ )
   {
      cin>>temp;
      if(!cin){
         cin.clear();
         cin.ignore (std::numeric_limits<std::streamsize>::max(), '\n');
         throw 20;
      }
      arr[i]=temp;

   }
 
   for ( j = 0 ; j < ( n - 1 ) ; j++ )
   {
      current_state = j; 
      for ( k = j + 1 ; k < n ; k++ )
      {
         if ( abs(arr[current_state]) > abs(arr[k]) )
            current_state = k;
      }
      if ( current_state != j )
      {
         swap = arr[j];
         arr[j] = arr[current_state];
         arr[current_state] = swap;
      }
   }
 
 cout<<"sorted array in the order of magnitude\n";
   for ( i = 0 ; i < n ; i++ )
      cout<<arr[i]<<"\t";
}
else
   cout<<"error in array size\n";
}
catch (int c)
{  
   cout<<"input error";
   //cin.clear();
}

 
cout<<"\npress 9 to stop/any no. to continue\n";
int stop;
cin>>stop;
if (stop==9)
   break;


}
 
   return 0;
}