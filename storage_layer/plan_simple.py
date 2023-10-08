class Operator:
    def __init__(self, id, est_rows, act_rows, task, acc_obj, exec_info, op_info, mem, disk):
        self.id = id
        self.est_rows = est_rows
        self.act_rows = act_rows
        self.task = task
        self.acc_obj = acc_obj
        self.exec_info = exec_info
        self.op_info = op_info
        self.mem = mem
        self.disk = disk
        self.children = []

        self.parent = None

    def contain_index_lookup(self):
        if self.is_index_lookup():
            return True
        for child in self.children:
            if child.contain_index_lookup():
                return True
        return False

    def contain_hash_join(self):
        if self.is_hash_join():
            return True
        for child in self.children:
            if child.contain_hash_join():
                return True
        return False

    def contain_hash_agg(self):
        if self.is_hash_agg():
            return True
        for child in self.children:
            if child.contain_hash_agg():
                return True
        return False

    def contain_sort(self):
        if self.is_sort():
            return True
        for child in self.children:
            if child.contain_sort():
                return True
        return False

    def is_db_operator(self):
        return self.task == "root"

    def is_kv_operator(self):
        return not self.is_db_operator()

    def is_hash_join(self):
        return "HashJoin" in self.id

    def is_hash_agg(self):
        return "HashAgg" in self.id

    def is_selection(self):
        return "Selection" in self.id

    def is_projection(self):
        return "Projection" in self.id

    def is_sort(self):
        return "Sort" in self.id

    def is_table_reader(self):
        return "TableReader" in self.id and self.is_db_operator()

    def is_table_scan(self):
        return "Table" in self.id and "Scan" in self.id and self.is_kv_operator()

    def is_index_reader(self):
        return "IndexReader" in self.id and self.is_db_operator()

    def is_index_scan(self):
        return "Index" in self.id and "Scan" in self.id and self.is_kv_operator()

    def is_index_lookup(self):
        return "IndexLookUp" in self.id and self.is_db_operator()

    def is_build_side(self):
        return "Build" in self.id

    def is_probe_side(self):
        return "Probe" in self.id

    def est_row_counts(self):
        return float(self.est_rows)

    def tidb_est_cost(self):
        return float(self.est_cost)

    def parse_fields(self, kv_str):
        # k1:v1, k2:v2, ...
        kv_map = {}
        for kv in kv_str.split(","):
            tmp = kv.split(":")
            if len(tmp) != 2:
                continue
            k, v = tmp[0].strip(), tmp[1].strip()
            kv_map[k] = v
        return kv_map

    def row_size(self):
        kv_map = self.parse_fields(self.op_info)
        return float(kv_map['row_size'])

    def batch_size(self):
        assert (self.is_index_lookup())
        kv_map = self.parse_fields(self.op_info)
        return float(kv_map['batch_size'])

    def exec_time_in_ms(self):
        # time:262.6µs, loops:1, Concurrency:OFF
        kv_map = self.parse_fields(self.exec_info)
        t = kv_map["time"]
        if t[-2:] == "ms":
            return float(t[:-2])
        elif t[-2:] == "µs":
            return float(t[:-2]) / 1000
        elif t[-2:] == "ns":
            return float(t[:-2]) / 1000000
        else:  # s
            return float(t[:-1]) * 1000

    def debug_print(self, indent):
        print("%s%s" % (indent, self.id))
        for child in self.children:
            child.debug_print(indent + "  ")


    def get_scan_range(self):
        range_scan=self.children
        assert "TableRangeScan" in range_scan.id



class Plan:
    def __init__(self, query, root):
        self.query = query
        self.root = root

    def debug_print(self):
        print("query: %s" % self.query)
        self.root.debug_print("")
        pass

    def exec_time_in_ms(self):
        return self.root.exec_time_in_ms()

    def tidb_est_cost(self):
        return self.root.tidb_est_cost()

    @staticmethod
    def format_op_id(id):
        for c in ['├', '─', '│', '└']:
            id = id.replace(c, ' ')
        num_prefix_spaces = 0
        for i in range(0, len(id)):
            if id[i] != ' ':
                num_prefix_spaces = i
                break
        return id.strip(" "), num_prefix_spaces

    @staticmethod
    def parse_plan(query, plan):
        plan = plan[1:]
        rows = [row.split("\t") for row in plan]
        operators = []
        prefix_spaces = []
        for i in range(len(rows)):
            r = rows[i]
            id, num_prefix_spaces = Plan.format_op_id(r[0])
            prefix_spaces.append(num_prefix_spaces)
            operators.append(Operator(id, r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8]))
            if i > 0:
                j = i - 1
                while j >= 0:
                    if prefix_spaces[j] < prefix_spaces[i]:
                        operators[j].children.append(operators[i])
                        break
                    j -= 1
        return Plan(query, operators[0])


if __name__ == '__main__':
    query = "SELECT * FROM imdb.title USE INDEX(primary) WHERE id>=2085515 AND id<=2085767"
    plan = [
      "id\testRows\tactRows\ttask\taccess object\texecution info\toperator info\tmemory\tdisk",
      "TableReader_6\t250.00\t\troot\t\ttime:10.1ms, loops:2, RU:3.323477, cop_task: {num: 2, max: 8.46ms, min: 1.52ms, avg: 4.99ms, p95: 8.46ms, max_proc_keys: 224, p95_proc_keys: 224, tot_proc: 7.01ms, tot_wait: 1.08ms, rpc_num: 2, rpc_time: 9.91ms, copr_cache: disabled, build_task_duration: 12.8µs, max_distsql_concurrency: 1}\tdata:TableRangeScan_5\t37.7 KB\tN/A",
      "└─TableRangeScan_5\t250.00\t\tcop[tikv]\ttable:title\ttikv_task:{proc max:4ms, min:4ms, avg: 4ms, p80:4ms, p95:4ms, iters:4, tasks:2}, scan_detail: {total_process_keys: 253, total_process_keys_size: 31977, total_keys: 255, get_snapshot_time: 834.6µs, rocksdb: {key_skipped_count: 253, block: {cache_hit_count: 6}}}\trange:[2085515,2085767], keep order:false, stats:pseudo\tN/A\tN/A"
    ]

    p = Plan.parse_plan(query, plan)
    print(p.exec_time_in_ms())
