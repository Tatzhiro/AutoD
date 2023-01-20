set -xu

NUM_TABLES=${5}
TABLE_SIZE=${6}
NUM_THREADS=${7}
TEST_DURATION=${8}
OLTP_WORK=${9}
INNO_BUF_SIZE=${10}
INNO_IO_CAP=${11}
FILENAME=${12}

mysql -h ${1} -u ${2} -p${3} -e "set global innodb_buffer_pool_size=${INNO_BUF_SIZE}; set global innodb_io_capacity=${INNO_IO_CAP}"
sysbench --db-driver=mysql --mysql-host=$1 --mysql-user=$2 --mysql-password=$3 --mysql-db=$4 --tables=${NUM_TABLES} --table_size=${TABLE_SIZE} --threads=${NUM_THREADS} --time=${TEST_DURATION} ${OLTP_WORK} run
python3 exporter.py --configs label=${OLTP_WORK} tablesize=${TABLE_SIZE} -f ${FILENAME}