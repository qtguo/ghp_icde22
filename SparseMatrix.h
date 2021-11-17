

#ifndef __SPARSEMATRIX_H__

	#define	__SPARSEMATRIX_H__

	#include <vector>
	#include <iostream>
	using namespace std;


	template<typename T>
	class SparseMatrix
	{

		public:

			// === CREATION ==============================================

			SparseMatrix(int n){
				this->construct(n, n);
			} // square matrix n×n

			SparseMatrix(int rows, int columns){
				this->construct(rows, columns);
			} // general matrix

			SparseMatrix(const SparseMatrix<T> & m){
				this->deepCopy(m);
			} // copy constructor

			SparseMatrix<T> & operator = (const SparseMatrix<T> & m){
				if (&m != this) {
					this->destruct();
					this->deepCopy(m);
				}

				return *this;
			}

			~SparseMatrix(void){
				this->destruct();
			}


			// === GETTERS / SETTERS ==============================================

			int getRowCount(void) const{
				return this->m;
			}
			int getColumnCount(void) const{
				return this->n;
			}



			// === VALUES ==============================================

			T get(int row, int col) const{
				this->validateCoordinates(row, col);

				int currCol;

				for (int pos = (*(this->rows))[row - 1] - 1; pos < (*(this->rows))[row] - 1; ++pos) {
					currCol = (*(this->cols))[pos];

					if (currCol == col) {
						return (*(this->vals))[pos];

					} else if (currCol > col) {
						break;
					}
				}

				return T();
			}

			SparseMatrix & set(T val, int row, int col){
				this->validateCoordinates(row, col);

				int pos = (*(this->rows))[row - 1] - 1;
				int currCol = 0;

				for (; pos < (*(this->rows))[row] - 1; pos++) {
					currCol = (*(this->cols))[pos];

					if (currCol >= col) {
						break;
					}
				}

				if (currCol != col) {
					if (!(val == T())) {
						this->insert(pos, row, col, val);
					}

				} else if (val == T()) {
					this->remove(pos, row);

				} else {
					(*(this->vals))[pos] = val;
				}

				return *this;
			}


			// === OPERATIONS ==============================================

			vector<T> multiply(const vector<T> & x) const{
				if (this->n != (int) x.size()) {
					cout<< "Cannot multiply: Matrix column count and vector size don't match." <<endl;
					//throw InvalidDimensionsException("Cannot multiply: Matrix column count and vector size don't match.");
				}

				vector<T> result(this->m, T());

				if (this->vals != NULL) { // only if any value set
					for (int i = 0; i < this->m; i++) {
						T sum = T();
						for (int j = (*(this->rows))[i]; j < (*(this->rows))[i + 1]; j++) {
							sum = sum + (*(this->vals))[j - 1] * x[(*(this->cols))[j - 1] - 1];
						}

						result[i] = sum;
					}
				}

				return result;
			}
			vector<T> operator * (const vector<T> & x) const{
				return this->multiply(x);
			}

			SparseMatrix<T> multiply(const SparseMatrix<T> & m) const{
				if (this->n != m.m) {
					cout<< "Cannot multiply: Matrix column count and vector size don't match." <<endl;
					//throw InvalidDimensionsException("Cannot multiply: Left matrix column count and right matrix row count don't match.");
				}

				SparseMatrix<T> result(this->m, m.n);

				T a;

				// TODO: more efficient?
				// @see http://www.math.tamu.edu/~srobertp/Courses/Math639_2014_Sp/CRSDescription/CRSStuff.pdf

				for (int i = 1; i <= this->m; i++) {
					for (int j = 1; j <= m.n; j++) {
						a = T();

						for (int k = 1; k <= this->n; k++) {
							a = a + this->get(i, k) * m.get(k, j);
						}

						result.set(a, i, j);
					}
				}

				return result;
			}
			SparseMatrix<T> operator * (const SparseMatrix<T> & m) const{
				return this->multiply(m);
			}

			SparseMatrix<T> add(const SparseMatrix<T> & m) const{
				if (this->m != m.m || this->n != m.n) {
					cout<< "Cannot multiply: Matrix column count and vector size don't match." <<endl;
					//throw InvalidDimensionsException("Cannot add: matrices dimensions don't match.");
				}

				SparseMatrix<T> result(this->m, this->n);

				// TODO: more efficient?
				// @see http://www.math.tamu.edu/~srobertp/Courses/Math639_2014_Sp/CRSDescription/CRSStuff.pdf

				for (int i = 1; i <= this->m; i++) {
					for (int j = 1; j <= this->n; j++) {
						result.set(this->get(i, j) + m.get(i, j), i, j);
					}
				}

				return result;
			}
			SparseMatrix<T> operator + (const SparseMatrix<T> & m) const{
				return this->add(m);
			}

			SparseMatrix<T> subtract(const SparseMatrix<T> & m) const{
				if (this->m != m.m || this->n != m.n) {
					cout<< "Cannot multiply: Matrix column count and vector size don't match." <<endl;
					//throw InvalidDimensionsException("Cannot subtract: matrices dimensions don't match.");
				}

				SparseMatrix<T> result(this->m, this->n);

				// TODO: more efficient?
				// @see http://www.math.tamu.edu/~srobertp/Courses/Math639_2014_Sp/CRSDescription/CRSStuff.pdf

				for (int i = 1; i <= this->m; i++) {
					for (int j = 1; j <= this->n; j++) {
						result.set(this->get(i, j) - m.get(i, j), i, j);
					}
				}

				return result;
			}
			SparseMatrix<T> operator - (const SparseMatrix<T> & m) const{
				return this->subtract(m);
			}


			// === FRIEND FUNCTIONS =========================================

			template<typename X>
			friend bool operator == (const SparseMatrix<X> & a, const SparseMatrix<X> & b){
				return ((a.vals == NULL && b.vals == NULL)
						|| (a.vals != NULL && b.vals != NULL && *(a.vals) == *(b.vals)))
					   && ((a.cols == NULL && b.cols == NULL)
						   || (a.cols != NULL && b.cols != NULL && *(a.cols) == *(b.cols)))
					   && *(a.rows) == *(b.rows);
			}

			template<typename X>
			friend bool operator != (const SparseMatrix<X> & a, const SparseMatrix<X> & b){
				return !(a == b);
			}

			template<typename X>
			friend ostream & operator << (ostream & os, const SparseMatrix<X> & matrix){
				for (int i = 1; i <= matrix.m; i++) {
					for (int j = 1; j <= matrix.n; j++) {
						if (j != 1) {
							os << " ";
						}

						os << matrix.get(i, j);
					}

					if (i < matrix.m) {
						os << endl;
					}
				}

				return os;
			}


		protected:

			int m, n;

			vector<T> * vals;
			vector<int> * rows, * cols;


			// === HELPERS / VALIDATORS ==============================================

			void construct(int rows, int columns){
				if (rows < 1 || columns < 1) {
					cout<< "Cannot multiply: Matrix column count and vector size don't match." <<endl;
					//throw InvalidDimensionsException("Matrix dimensions cannot be zero or negative.");
				}

				this->m = rows;
				this->n = columns;

				this->vals = NULL;
				this->cols = NULL;
				this->rows = new vector<int>(rows + 1, 1);
			}

			void destruct(void){
				if (this->vals != NULL) {
					delete this->vals;
					delete this->cols;
				}

				delete this->rows;
			}

			void deepCopy(const SparseMatrix<T> & matrix){
				this->m = matrix.m;
				this->n = matrix.n;
				this->rows = new vector<int>(*(matrix.rows));

				if (matrix.vals != NULL) {
					this->cols = new vector<int>(*(matrix.cols));
					this->vals = new vector<T>(*(matrix.vals));
				}
			}

			void validateCoordinates(int row, int col) const{
				if (row < 1 || col < 1 || row > this->m || col > this->n) {
					cout<< "Coordinates out of range." <<endl;
					// throw InvalidCoordinatesException("Coordinates out of range.");
				}
			}

			void insert(int index, int row, int col, T val){
				if (this->vals == NULL) {
					this->vals = new vector<T>(1, val);
					this->cols = new vector<int>(1, col);

				} else {
					this->vals->insert(this->vals->begin() + index, val);
					this->cols->insert(this->cols->begin() + index, col);
				}

				for (int i = row; i <= this->m; i++) {
					(*(this->rows))[i] += 1;
				}
			}
			void remove(int index, int row){
				this->vals->erase(this->vals->begin() + index);
				this->cols->erase(this->cols->begin() + index);

				for (int i = row; i <= this->m; i++) {
					(*(this->rows))[i] -= 1;
				}
			}

	};

#endif
