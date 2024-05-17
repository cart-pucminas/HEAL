import psycopg2
import psycopg2.extras

from config.ConfigCreateDatabase import config


class DatabaseUtil:

    __instance = None

    def __init__(self):
        self.connection = psycopg2.connect(
            f'dbname={config.DATASOURCE_DB} user={config.DATASOURCE_USERNAME} password={config.DATASOURCE_PASSWORD}')
        self.cursor = self.connection.cursor(
            cursor_factory=psycopg2.extras.DictCursor)

    def _close_connection(self):
        if(self.connection):
            self.cursor.close()
            self.connection.close()

    def save_or_update(self, string_db, params):
        try:
            #conn, cur = self._conexao_db()
            self.cursor.execute(string_db, params)
            self.connection.commit()
        except psycopg2.Error as error:
            print("Failed to read data from table")
            print(error)
            self.connection.rollback()
            return False
        finally:
            # self._close_connection()
            return True

    def delete_all(self, string_db):
        try:
            #conn, cur = self._conexao_db()
            self.cursor.execute(string_db)
            self.connection.commit()
        except psycopg2.Error as error:
            print("Failed to read data from table")
            print(error)
            self.connection.rollback()
            return False
        finally:
            # self._close_connection()
            return True

    def findAll(self, string_db, params=None):
        try:
            #conn, cur = self._conexao_db()
            self.cursor.execute(string_db, params)
            return self.cursor.fetchall()
        except psycopg2.Error as error:
            print("Failed to read data from table")
            print(error)
            return None
        # finally:
        #     # self._close_connection()
        #     return True

    def findOne(self, string_db, params):
        try:
            #conn, cur = self._conexao_db()
            self.cursor.execute(string_db, params)
            return self.cursor.fetchone()
        except psycopg2.Error as error:
            print("Failed to read data from table")
            print(error)
            return None
        # finally:
        #     # self._close_connection()
        #     return True
