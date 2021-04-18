# STANDARD LIBARIES
import sqlite3
from datetime import datetime, date, timedelta

# THIRD PARTY LIBARIES


# LOCAL LIBARIES
from definitions.classes.blacklist_entry import BlacklistEntry
from config.program_config import BLACKLIST_FILE, INFO, ERROR, WARNING
from helper.logger import log
from config import configuration as conf


last_updated = None

def create_connection():
    global last_updated

    try:
        conn = sqlite3.connect(BLACKLIST_FILE)
        log(INFO, "Database connection created.")
    except sqlite3.Error as e:
        log(ERROR, "Database connection could not been established.")
        log(ERROR, str(e))
        return None

    today = date.today().strftime("%d/%m/%Y")
    if last_updated is None:

        if conf.check_for_element("Blacklist", "Last Updated"):
            last_updated = conf.get_element("Blacklist", "Last Updated")

            if last_updated.__ne__(today):
                c = conn.cursor()
                last_updated_date = datetime.strptime(last_updated, "%d/%m/%Y")
                update_db(conn, last_updated_date, datetime.strptime(today, "%d/%m/%Y"))
                c.close()
                last_updated = today
                conf.set_element("Blacklist", "Last Updated", today)

        else:
            conf.add_section("Blacklist")
            conf.add_element("Blacklist", "Last Updated", today)
    else:
        if last_updated.__ne__(today):
            last_updated_date = datetime.strptime(last_updated, "%d/%m/%Y")
            update_db(conn, last_updated_date, datetime.strptime(today, "%d/%m/%Y"))


    return conn


def create_table(conn):
    c = conn.cursor()
    c.execute('''CREATE TABLE blacklist(domainname text, not_after text)''')
    conn.commit()
    log(INFO, "Table 'blacklist' created in sqlite database.")
    return


def add_entry(entry: BlacklistEntry):
    conn = create_connection()
    c = conn.cursor()
    c.execute(''' SELECT name FROM sqlite_master WHERE type='table' AND name='blacklist'; ''')
    data = (entry.domainname, entry.not_after)
    conn.commit()

    if c.fetchone() is None:
        create_table(conn)

    exists = check_for_entry(entry.domainname)

    if not exists:
        sql = ''' INSERT INTO blacklist(domainname,not_after)
                      VALUES(?,?) '''
        c.execute(sql, data)
        conn.commit()
        log(INFO, "{} inserted into the blacklist.".format(str(entry.domainname)))
    else:
        log(INFO, "{} is still in the blacklist.".format(str(entry.domainname)))

    conn.close()

    return


def update_db(conn, last_updated_date, today):
    delta = today - last_updated_date

    for i in range(delta.days + 1):
        c = conn.cursor()
        day = last_updated_date + timedelta(days=i)
        execs = c.execute(''' DELETE from blacklist WHERE not_after=? ''', (day,))
        conn.commit()
        log(INFO, "Database updated. Deleted records: {}".format(execs.rowcount))

    return


def check_for_entry(domainname):
    conn = create_connection()
    c = conn.cursor()
    c.execute("SELECT * FROM blacklist WHERE domainname=?", (domainname,))

    row = c.fetchone()

    if row:
        not_after = row[1]
        return (domainname, not_after)

    return None

