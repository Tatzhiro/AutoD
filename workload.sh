set -xu

NUM_TABLES=${5}
TABLE_SIZE=${6}
NUM_THREADS=${7}
TEST_DURATION=${8}
OLTP_WORK=${9}
INNO_BUF_SIZE=${10}
INNO_IO_CAP=${11}
FILENAME=${12}


sysbench --db-driver=mysql --mysql-host=$1 --mysql-user=$2 --mysql-password=$3 --mysql-db=$4 --tables=${NUM_TABLES} --table_size=${TABLE_SIZE} --threads=${NUM_THREADS} --time=${TEST_DURATION} ${OLTP_WORK} run
python3 exporter.py -f ${FILENAME}
echo "" >> ${FILENAME}
echo "num_tables,table_size,num_threads,test_duration,workload,innodb_buffer_pool_size,innodb_io_capacity" >> ${FILENAME}
echo "${NUM_TABLES},${TABLE_SIZE},${NUM_THREADS},${TEST_DURATION},${OLTP_WORK},${INNO_BUF_SIZE},${INNO_IO_CAP}" >> ${FILENAME}