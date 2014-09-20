#include <iostream> 
#include <stdlib.h> 
#include <limits>
using namespace std; 

int julian(int year, int month, int day) { 
  int a = (14 - month) / 12; 
  int y = year + 4800 - a; 
  int m = month + 12 * a - 3; 
  if (year > 1582 || (year == 1582 && month > 10) || (year == 1582 && month == 10 && day >= 15)) 
    return day + (153 * m + 2) / 5 + 365 * y + y / 4 - y / 100 + y / 400 - 32045; 
  else 
    return day + (153 * m + 2) / 5 + 365 * y + y / 4 - 32083; 
} 

int main() { 
  int y,m,d,total_days,j1,j2;

int a[12]={31,28,31,30,31,30,31,31,30,31,30,31},b[12]={31,29,31,30,31,30,31,31,30,31,30,31};
  while(1)
  {
  cout << "enter date month year\n"; 
  cin >> d; 
  cin >> m; 
  cin >> y; 
           if(!cin){
                   cin.clear();
                   cin.ignore (std::numeric_limits<std::streamsize>::max(), '\n');
                   cout<<"not an integer";
                }
      
  else
      { bool leap;
                        if ( y%400 == 0)
                            leap=true;
                          else if ( y%100 == 0)
                            leap=false;
                          else if ( y%4 == 0 )
                            leap=true;
                          else
                            leap=false;  
if (leap && (d>b[m-1]||m>12))
    { cout<<"input error";}
else if (d>a[m-1]||m>12)
      cout<< "input error";
else
          {

          j1 = julian(y, 1, 1); 
          j2 = julian(y, m,d); 
          total_days = abs(j2 - j1);
           
          cout << "no of days are \t" << total_days << endl; 
           }
         }
  int stop;
cout<<"\npress 9 to stop/any no. to continue\n";
cin>>stop;
if (stop==9)
   break;
}
  return 0; 
}