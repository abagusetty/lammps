// C++ program to demonstrate working of
// Variadic function Template
#include <iostream>
#include <string>
#include <utility>
using namespace std;
 
// // To handle base case of below recursive
// // Variadic function Template
void print(int a, int b, double x, std::string f)
{
  cout << "I am empty function and "
    "I am called at last.\n";
}
 
// Variadic function Template that takes
// variable number of arguments and prints
// all of them.
template <typename... Types>
void print(Types&&... var2)
{
  //cout << var1 << endl;
 
  print(std::forward<Types>(var2)...);
}
 
// Driver code
int main()
{
    print(1, 2, 3.14, "Pass me any ");
    return 0;
}
