import mysql.connector
import pandas as pd
import time


if __name__ == '__main__':
	user = input("Enter database username: ")
	password = input("Enter database password for " + user + ": ")
	table =  input("Enter ticker: ")
	connection = mysql.connector.connect(host = 'localhost', database = 'equities_by_ticker', user = user, password = password)
	t =  time.time()
	df = pd.read_sql('SELECT EPS Growth FROM ' + table, con = connection)
	print(df)
	print("Time taken for sql for " + table + ": " + str(time.time() - t))
	t = time.time()
	df = pd.read_csv(table + ".csv", index_col = 0)['EPS Growth']
	print(df)
	print("Time taken for csv for " + table + ": " + str(time.time() - t))
