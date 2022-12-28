# missing detection
# import missingno as msno
# data = pred
# missings = data.isna().sum()
# print(missings.loc[missings > 0])
# print(data[["date", "ticker"]].drop_duplicates())
# msno.matrix(data)
# plt.savefig('../../venv/img/Missings.png', dpi=400)

# sector_PE_means = pred[(pred["PE"]>0) & (pred["PE"]<max_PE)].groupby('sector')['PE'].mean().rename("sector_PE")
# industry_PE_means = pred[(pred["PE"]>0) & (pred["PE"]<max_PE)].groupby('industry')['PE'].mean().rename("industry_PE")
# pred = pred.join(sector_PE_means, on="sector")
# pred = pred.join(industry_PE_means, on="industry")
#
#
# pred_cat=pred[['sector', 'industry']]
# le = {}
# for col in cat_cols:
#     le[col] = LabelEncoder()
#     pred_cat[col] = le[col].fit_transform(pred_cat[col])
# ohe = OneHotEncoder(sparse=False)
# ohe.fit(pred_cat)
# cat_ind_names = []
# for c in cat_cols:
#     cat_ind_names = cat_ind_names + [c + "_" + str.replace(cls," ","_") for cls in le[c].classes_ ]