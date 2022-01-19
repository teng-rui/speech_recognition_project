import sidekit


cep = None


# GMM model, 1024 features
ubm = sidekit.Mixture()
llk = ubm.EM_uniform(cep, distrib_nb=400)


train_idmap = sidekit.IdMap()
devel_idmap = sidekit.IdMap()
test_idmap = sidekit.IdMap()

train_stat = sidekit.StatServer(statserver_file_name=train_idmap, ubm=ubm)
devel_stat = sidekit.StatServer(statserver_file_name=devel_idmap, ubm=ubm)
test_stat = sidekit.StatServer(statserver_file_name=test_idmap, ubm=ubm)


# TV matrix
fa = sidekit.FactorAnalyser()
fa.total_variability_single(stat_server_filename=train_stat, ubm=ubm, tv_rank=400, nb_iter=10)

train_iv = fa.extract_ivectors_single(ubm=ubm, stat_server=train_stat)
devel_iv = fa.extract_ivectors_single(ubm=ubm, stat_server=devel_stat)
test_iv = fa.extract_ivectors_single(ubm=ubm, stat_server=test_stat)
