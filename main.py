__author__ = 'Rahul Penti'
import random

def main():
    global pro,  man,  exp,  log,  imp,  distij,  demand,  distjk,  distli,  distlj,  distlk,  lstock,  linv_c,  binv_c,  pop_size, gen_loc
    gen_loc = 3
    pop_size = 10
    pro, man, exp, log, imp = 5, 3, 2, 3, 2
    distij = [[42, 93, 93], [39, 57, 66], [66, 12, 39], [99, 33, 54], [45, 54, 30]]
    distjk = [[639, 740], [640, 772], [702, 825]]
    distli = [[15, 45, 96, 129, 93], [66, 48, 15, 45, 45], [111, 90, 69, 75, 30]]
    distlj = [[51, 99, 102], [57, 15, 24], [75, 63, 33]]
    distlk = [[602, 710], [644, 774], [723, 850]]
    lstock = [[88, 97, 107, 105, 122, 130], [132, 145, 161, 158, 183, 196], [85, 94, 104, 102, 118, 127], [41, 45, 50, 49, 50, 60], [201, 222, 246, 240, 279, 299], [547, 603, 668, 654, 758, 812]]
    demand = [67092, 74311, 80720, 79010, 90692, 96856]
    linv_c = 22.05
    binv_c = 105
    tran_rent1 = [40, 70]
    tran_rent2 = [240, 420]
    genetic_algo()

def genetic_algo():
    rand_pop = init_pop()
    children = offspring(rand_pop)
    

def init_pop():
    li_trans,  li_truck, li_type = [[[[0 for x in range(man)] for y in range(pro)]for z in range(6)]for l in range(pop_size)],  [[[[0 for x in range(man)] for y in range(pro)]for z in range(6)]for l in range(pop_size)], [[[[0 for x in range(man)] for y in range(pro)]for z in range(6)]for l in range(pop_size)]
    inter_trans, inter_truck, inter_type = [[[[0 for x in range(exp)] for x in range(man)]for x in range(6)]for x in range(pop_size)],  [[[[0 for x in range(exp)] for x in range(man)]for x in range(6)]for x in range(pop_size)], [[[[0 for x in range(exp)] for x in range(man)]for x in range(6)]for x in range(pop_size)]
    del_trans = [[[[0 for x in range(imp)] for x in range(exp)]for x in range(6)]for x in range(pop_size)]
    for i in range(0,  pop_size):
        for j in range(0,  6):
            for k in range(0,  pro):
                for l in range(0, man):
                    li_trans[i][j][k][l] =random.randint(10,  900)
                    li_truck[i][j][k][l] =random.randint(2,  100)
                    li_type[i][j][k][l] =random.randint(0, 3)
            for k in range(0,  man):
                for l in range(0,  exp):
                    inter_trans[i][j][k][l] =random.randint(10,  900)
                    inter_truck[i][j][k][l] =random.randint(2,  100)
                    inter_type[i][j][k][l] =random.randint(0,  3)
            for k in range(0,  exp):
                for l in range(0,  imp):
                    del_trans[i][j][k][l] =random.randint(10,  900)

    return li_trans, li_truck, inter_trans, inter_truck,  del_trans, li_type, inter_type

def offspring(rand_pop):
    chrom_set = make_chrom(rand_pop)
    cross_set = cross_1(chrom_set)
    deco_chrom = decode_chrom(cross_set)
    fitn_1 = fitness(deco_chrom)
    print fitn_1
    #local_search = loc_ser(cross_set)
    #decode = decode_chrom(cross_set)
    #local_search = fitness(rand_pop)

def make_chrom(rand_pop):
    li_chrom, li_trans_chrom, inter_chrom, inter_trans_chrom, del_chrom = [[]for x in range(pop_size)],  [[] for y in range(pop_size)],  [[] for y in range(pop_size)],  [[] for y in range(pop_size)],  [[] for y in range(pop_size)]
    for i in range(0,  pop_size):
        for j in range(0,  6):
            for k in range(0,  pro):
                for l in range(0, man):
                    li_chrom[i].append(rand_pop[0][i][j][k][l])
                    li_trans_chrom[i].append(rand_pop[1][i][j][k][l])

        for j in range(0,  6):
            for k in range(0,  man):
                for l in range(0,  exp):
                    inter_chrom[i].append(rand_pop[2][i][j][k][l])
                    inter_trans_chrom[i].append(rand_pop[3][i][j][k][l])
        for j in range(0,  6):    
            for k in range(0,  exp):
                for l in range(0,  imp):
                    del_chrom[i].append(rand_pop[4][i][j][k][l])
    return li_chrom, li_trans_chrom, inter_chrom, inter_trans_chrom, del_chrom

def cross_1(set):
    for i in range(len(set)):
        for j in range(len(set[i])):
            rand_seq = range(len(set[i][j]))
            random.shuffle(rand_seq)
            for k in range(len(rand_seq)/2):
                n_1, n_2 = rand_seq[k], rand_seq[len(rand_seq)-k-1]
                if i in [0, 2, 4]:
                    p_1, p_2 = bin(set[i][j][n_1])[2:].zfill(11), bin(set[i][j][n_2])[2:].zfill(11)
                if i in [1, 3]:
                    p_1, p_2 = bin(set[i][j][n_1])[2:].zfill(8), bin(set[i][j][n_2])[2:].zfill(8)
                temp1 = []
                temp2 = []
                ran_num = random.sample(xrange(1, len(p_1)-2), 2)
                r_n1, r_n2 = ran_num[0], ran_num[1]
                if r_n1 > r_n2:
                    r_n1, r_n2 = r_n2, r_n1
                for l in range(r_n1,r_n2):
                    p_2, p_1 = p_2[0:l] + str(p_1[l]) + p_2[l+1:], p_1[0:l] +  str(p_2[l]) + p_1[l+1:]
                set[i][j][n_1], set[i][j][n_2] = int(p_1,2), int(p_2,2)
    return mut_1(set)
def mut_1(set):
    for i in range(len(set)):
        for j in range(len(set[i])):
            ma1 = max(set[i][j])
            ma2 = min(set[i][j])
            for k in range(len(set[i][j])):
                p_1 = set[i][j][k]
                ran_num = random.random()
                if ran_num < 50:
                    mod = -1 + (2*ran_num)**(1/1.5)
                else:
                    mod = 1 - (2*(1-ran_num))**(1/1.5)
                mut_val = p_1 + (ma1 - ma2)*mod

                set[i][j][k] = round(mut_val,2)
                #print set[i][j][k]
    return set
def decode_chrom(set):
    li_trans,  li_truck, li_type = [[[[0 for x in range(man)] for y in range(pro)]for z in range(6)]for l in range(pop_size)],  [[[[0 for x in range(man)] for y in range(pro)]for z in range(6)]for l in range(pop_size)], [[[[0 for x in range(man)] for y in range(pro)]for z in range(6)]for l in range(pop_size)]
    inter_trans, inter_truck, inter_type = [[[[0 for x in range(exp)] for x in range(man)]for x in range(6)]for x in range(pop_size)],  [[[[0 for x in range(exp)] for x in range(man)]for x in range(6)]for x in range(pop_size)], [[[[0 for x in range(exp)] for x in range(man)]for x in range(6)]for x in range(pop_size)]
    del_trans = [[[[0 for x in range(imp)] for x in range(exp)]for x in range(6)]for x in range(pop_size)]
    for i in range(0,  pop_size):
        for j in range(0,  6):
            for k in range(0,  pro):
                for l in range(0, man):
                    li_trans[i][j][k][l] =set[0][i].pop(0)
                    li_truck[i][j][k][l] =set[1][i].pop(0)
            for k in range(0,  man):
                for l in range(0,  exp):
                    inter_trans[i][j][k][l] =set[2][i].pop(0)
                    inter_truck[i][j][k][l] =set[3][i].pop(0)
                    #inter_type[i][j][k][l] =random.randint(0,  1)
            for k in range(0,  exp):
                for l in range(0,  imp):
                    del_trans[i][j][k][l] =set[4][i].pop()

    return li_trans, li_truck, inter_trans, inter_truck,  del_trans, li_type, inter_type
def fitness(set):
    global inv_avl
    inv_avl = inv_cal(set)
    ex_1 = [0 for x in range(0, len(set[0]))]
    for h in range(0, len(set[0])):
        con_sub = 0
        for i in range(0,6):  ##LIVE STOCK CONSTRAINT
            num_qw = 0
            for j in range(0,len(set[0][h][i])):
                num_li = 0
                for k in range(0,len(set[0][h][i][j])):
                    num_li = num_li + set[0][h][i][j][k]
                if num_li > lstock[i][j]:
                    num_qw = num_qw - lstock[i][j] + num_li
            con_sub = num_qw + con_sub
        ex_1[h] = ex_1[h] + con_sub
    ex_2 = [0 for x in range(0, len(set[0]))]
    imp_q = [[[0 for x in range(0,len(set[0][0][0]))]for y in range(0,6)]for x in range(0, len(set[0]))]
    exp_q = [[[0 for x in range(0,len(set[2][0][0]))]for y in range(0,6)]for x in range(0, len(set[0]))]
    for h in range(0, len(set[0])):
        for i in range(0,6):  ##Live stock Inventory
            for j in range(0,len(set[2][h][i])):
                for k in range(0,len(set[2][h][i][j])):
                    exp_q[h][i][j] = exp_q[h][i][j] + set[2][h][i][j][k]
            for k in range(0,len(set[0][h][i][0])):
                for l in range(0,len(set[0][h][i])):
                    #print k, l
                    imp_q[h][i][l] = imp_q[h][i][l] + set[0][h][i][l][k]
        for i in range(0,6):
            for j in range(len(exp_q[h][i])):
                if i == 0:
                        if inv_avl[1][h][i][j] + imp_q[h][i][j] < exp_q[h][i][j]:
                            ex_2[h] = ex_2[h] - inv_avl[1][h][i][j] + exp_q[h][i][j] - imp_q[h][i][j]
                else:
                        if inv_avl[1][h][i][j] + inv_avl[1][h][i-1][j] + imp_q[h][i][j] < exp_q[h][i][j]:
                            ex_2[h] = ex_2[h] - inv_avl[1][h][i][j] + exp_q[h][i][j] - imp_q[h][i][j] - inv_avl[1][h][i-1][j]
    ex_3 = [0 for x in range(0, len(set[0]))]
    for h in range(0,len(set[0])):
        for i in range(0,6):  ##Demand CONSTRAINT
            con_sub = 0
            for j in range(0,len(set[4][h][i])):
                for k in range(0,len(set[4][h][i][j])):
                    con_sub = con_sub + set[4][h][i][j][k]
            print con_sub, demand[i]
            if con_sub > demand[i]:
                ex_3[h] = ex_3[h] + (con_sub - demand[i])
    ex_4 = [0 for x in range(0, len(set[0]))]
    for h in range(0,len(set[0])):
        for i in range(0,4):
            dem_t = 0
            pen_1 = 0
            for j in range(0,len(set[2][h][i])):
                for k in range(0,len(set[2][h][i][j])):
                    for l in range(0,2):
                        dem_t = dem_t + set[2][h][i+l][j][k]
                pen_1 = pen_1 + inv_avl[1][h][i][j] - dem_t
            ex_4[h] = ex_4[h] + pen_1
    ex_5 = [0 for x in range(0, len(set[0]))]
    imp_q = [[[0 for x in range(0,len(set[2][0][0]))]for y in range(0,6)]for x in range(0, len(set[0]))]
    exp_q = [[[0 for x in range(0,len(set[4][0][0]))]for y in range(0,6)]for x in range(0, len(set[0]))]
    for h in range(0, len(set[0])):
        for i in range(0,6):  ##Live stock Inventory
            for j in range(0,len(set[4][h][i])):
                for k in range(0,len(set[4][h][i][j])):
                    exp_q[h][i][j] = exp_q[h][i][j] + set[4][h][i][j][k]
            for k in range(0,len(set[2][h][i][0])):
                for l in range(0,len(set[2][h][i])):
                    #print k, l
                    imp_q[h][i][l] = imp_q[h][i][l] + set[2][h][i][l][k]
        for i in range(0,6):
            for j in range(len(exp_q[h][i])):
                if i == 0:
                    if inv_avl[2][h][i][j] + imp_q[h][i][j] < exp_q[h][i][j]:
                        ex_5[h] = ex_5[h] + inv_avl[2][h][i][j] - exp_q[h][i][j] + imp_q[h][i][j]
                else:
                    if inv_avl[2][h][i][j] + imp_q[h][i][j]+inv_avl[1][h][i-1][j] < exp_q[h][i][j]:
                        ex_5[h] = ex_5[h] + inv_avl[2][h][i][j] - exp_q[h][i][j] + imp_q[h][i][j] + inv_avl[1][h][i-1][j]
    const_all = [0 for x in range(0, len(set[0]))]
    for h in range(0, len(ex_1)):
        const_all[h] = ex_1[h] + ex_2[h] + ex_3[h] + ex_4[h] + ex_5[h]
        #print ex_1[h], ex_2[h], ex_3[h], ex_4[h], ex_5[h]
    #print len(const_all)
    return const_all







def inv_cal(set):
    li_inv = [[[0 for y in range(pro)]for z in range(6)]for l in range(pop_size)]
    inter_inv = [[[0 for x in range(man)]for x in range(6)]for x in range(pop_size)]
    del_inv = [[[0 for x in range(exp)]for x in range(6)]for x in range(pop_size)]
    for i in range(0,  pop_size):
        for j in range(0,  6):
            for k in range(0,  pro):
                used = 0
                for l in range(0, man):
                    #print type(set[0][i][j][k]), type(used)
                    used = set[0][i][j][k][l] + used
                li_inv[i][j][k] = lstock[j][k] - used
            for k in range(0,  man):
                used = 0
                for l in range(0, imp):
                    used = set[2][i][j][k][l] + used
                inter_inv[i][j][k] = lstock[j][k]  - used
            for k in range(0,  imp):
                used = 0
                for l in range(0, exp):
                    used = set[4][i][j][k][l] + used
                del_inv[i][j][k] = lstock[j][k]  - used
    return li_inv, inter_inv, del_inv

def truck_empty(set):
    li_full_truck   = [[[[0 for x in range(man)] for y in range(pro)]for z in range(6)]for l in range(pop_size)]
    inter_full_truck = [[[[0 for x in range(exp)] for x in range(man)]for x in range(6)]for x in range(pop_size)]
    li_inc_quant = [[[[0 for x in range(man)] for y in range(pro)]for z in range(6)]for l in range(pop_size)]
    inter_inc_quant = [[[[0 for x in range(exp)] for x in range(man)]for x in range(6)]for x in range(pop_size)]
    li_inc_mat = [[[[0 for x in range(man)] for y in range(pro)]for z in range(6)]for l in range(pop_size)]
    inter_inc_mat = [[[[0 for x in range(exp)] for x in range(man)]for x in range(6)]for x in range(pop_size)]
    for i in range(0,  pop_size):
        for j in range(0,  6):
            for k in range(0,  pro):
                for l in range(0, man):
                    li_full_truck[i][j][k][l] = round(set[0][i][j][k][l]/20, 0)
                    li_inc_quant[i][j][k][l] = set[0][i][j][k][l] - 20*li_full_truck[i][j][k][l]
                    if li_inc_quant > 0:
                        li_inc_mat[i][j][k][l] = 1
            for k in range(0,  man):
                for l in range(0,  exp):
                    inter_full_truck[i][j][k][l] =round(set[3][i][j][k][l]/20, 0)
                    inter_inc_quant[i][j][k][l] = set[2][i][j][k][l] - 20*inter_full_truck[i][j][k][l]
                    if inter_inc_quant > 0:
                        inter_inc_mat[i][j][k][l] = 1
    return li_inc_mat, li_inc_quant, li_full_truck, inter_inc_mat, inter_inc_quant, inter_full_truck











def loc_ser(set):
    loc_res = [[]for x in range(pop_size)]
    org_chrom = decode_chrom(set)
    global inv_avl
    inv_avl = inv_cal(org_chrom)
    org_fit = fitness(org_chrom)
    #print org_chrom[1][8][5][1]
    for j in range(len(set[0])):
        #print len(set[i])
        loc_res[j] = ind_chrom(org_chrom,j)
        for l in range(0,gen_loc):
            upd_chrom = mut_loc(loc_res[j])
            upd_fit = fitness(loc_res[j])
            if upd_fit < org_fit:
                loc_res[j] = upd_chrom

    return loc_res
def ind_chrom(set,x):
    li_trans,  li_truck, li_type = [[[[0 for x in range(man)] for y in range(pro)]for z in range(6)]for l in range(pop_size)],  [[[[0 for x in range(man)] for y in range(pro)]for z in range(6)]for l in range(pop_size)], [[[[0 for x in range(man)] for y in range(pro)]for z in range(6)]for l in range(pop_size)]
    inter_trans, inter_truck, inter_type = [[[[0 for x in range(exp)] for x in range(man)]for x in range(6)]for x in range(pop_size)],  [[[[0 for x in range(exp)] for x in range(man)]for x in range(6)]for x in range(pop_size)], [[[[0 for x in range(exp)] for x in range(man)]for x in range(6)]for x in range(pop_size)]
    del_trans = [[[[0 for x in range(imp)] for x in range(exp)]for x in range(6)]for x in range(pop_size)]
    for j in range(0,  6):
            for k in range(0,  pro):
                for l in range(0, man):
                    li_trans[0][j][k][l] = set[0][x][j][k][l]
                    li_truck[0][j][k][l] = set[1][x][j][k][l]
            for k in range(0,  man):
                for l in range(0,  exp):
                    inter_trans[0][j][k][l] = set[2][x][j][k][l]
                    inter_truck[0][j][k][l] = set[3][x][j][k][l]
                    #inter_type[i][j][k][l] =random.randint(0,  1)
            for k in range(0,  exp):
                for l in range(0,  imp):
                    del_trans[0][j][k][l] = set[4][x][j][k][l]

    return li_trans, li_truck, inter_trans, inter_truck,  del_trans, li_type, inter_type





def mut_loc(set):
    for i in range(0,len(set[0])):
        for j in range(0,len(set[0][i])):
            for k in range(0, len(set[0][i][j])):
                for l in range(0, len(set[0][i][j][k])):
                    z, s = random.random(), random.random()
                    if z > 0.8:
                        if s > 0.5:
                            set[0][i][j][k][l] = set[0][i][j][k][l] + random.randint(int(min(set[0][i][j][k])), int(max(set[0][i][j][k])))
                        else:
                            set[0][i][j][k][l] = set[0][i][j][k][l] + random.randint(int(min(set[0][i][j][k])), int(max(set[0][i][j][k])))
    return set

main()


