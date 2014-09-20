#include <stdio.h>
 #include <iostream>
#include <stdlib.h>
#include <limits>
using namespace std;
int main()
{
   int n;
 
   printf("array size\n");
   cin>>n;
   int array[n], search, c;
 
   printf("enter elements\n");
 
   for (c = 0; c < n; c++)
      scanf("%d", &array[c]);
 while(1){
   printf("search an element\n");
   cin>>search;
   if(!cin){
         cin.clear();
         cin.ignore (std::numeric_limits<std::streamsize>::max(), '\n');
         cout<<"not an integer";

      }
   else{
   for (c = 0; c < n; c++)
   {
      if (array[c] == search) 
      {
         printf("%d is at  %d\n", search, c+1);
         break;
      }
   }
   if (c == n)
      printf("not present\n");
}
 int stop;
cout<<"\npress 9 to stop/any no. to continue\n";
cin>>stop;
if (stop==9)
   break;

 }
   return 0;
}