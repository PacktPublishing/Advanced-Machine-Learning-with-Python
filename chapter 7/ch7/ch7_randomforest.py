from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_jobs = 3, verbose = 3, n_estimators=20)
rf.fit(DisruptionInformation_train.targets,DisruptionInformation_train.data)

r2 = r2_score(DisruptionInformation.data, rf.predict(DisruptionInformation.targets))
mse = np.mean((DisruptionInformation.data - rf.predict(DisruptionInformation.targets))**2)

pl.scatter(DisruptionInformation.data, rf.predict(DisruptionInformation.targets))
pl.plot(np.arange(8, 15), np.arange(8, 15), label="r^2=" + str(r2), c="r")
pl.legend(loc="lower right")
pl.title("RandomForest Regression with scikit-learn")
pl.show()
