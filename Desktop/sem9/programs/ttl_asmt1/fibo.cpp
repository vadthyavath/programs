#include <stdio.h>
 #include <iostream>
#include <stdlib.h>
#include <limits>
using namespace std;
 
int main()
{ while(1){
   int n, first = 0, second = 1, next, c;
 
   printf("enter n\n");
   cin>>n;
 
 
   printf("%dterms are \n",abs(n));
 
   for ( c = 0 ; c < abs(n) ; c++ )
   {
      if ( c <= 1 )
         next = c;
      else
      {
         next = first + second;
         first = second;
         second = next;
      }
      printf("%d\t",next);
   }

int stop;
cout<<"\npress 9 to stop/any no. to continue\n";
cin>>stop;
if (stop==9)
   break;
}
   return 0;
}