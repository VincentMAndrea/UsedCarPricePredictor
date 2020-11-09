import joblib
import pandas as pd
from tkinter import *

root = Tk()
root.title('Quick Book Look')
root.iconbitmap('car_23773.ico')

model = joblib.load('vehicle_value_model_alt_est10.pkl')
df = pd.read_csv('vehicles.csv', usecols=['Year', 'Make', 'Model', 'Condition', 'Title', 'Transmission', 'Odometer', 'Price'])
top_1000 = [x for x in df.Model.value_counts().sort_values(ascending=False).head(1000).index]
df_1000 = df[df.Model.isin(top_1000)]
df_final = pd.get_dummies(df_1000)

year_in_label = Label(root, text='Vehicle Year:').grid(row=0)
make_in_label = Label(root, text='Vehicle Make:').grid(row=1)
model_in_label = Label(root, text='Vehicle Model:').grid(row=2)
condition_in_label = Label(root, text='Vehicle Condition:').grid(row=3)
title_in_label = Label(root, text='Title Status:').grid(row=4)
transmission_in_label = Label(root, text='Transmission Type:').grid(row=5)
odometer_in_label = Label(root, text='Odometer(mi):').grid(row=6)

e_year = Entry(root)
e_year.grid(row=0, column=1)
e_make = Entry(root)
e_make.grid(row=1, column=1)
e_model = Entry(root)
e_model.grid(row=2, column=1)
e_condition = Entry(root)
e_condition.grid(row=3, column=1)
e_title = Entry(root)
e_title.grid(row=4, column=1)
e_transmission = Entry(root)
e_transmission.grid(row=5, column=1)
e_odometer = Entry(root)
e_odometer.grid(row=6, column=1)


def match_dummies(args):
    args = pd.get_dummies(args)
    missing_cols = set(df_final.columns) - set(args.columns)
    for c in missing_cols:
        args[c] = 0
    args = args[df_final.columns]
    args = args.loc[:, args.columns != 'Price']
    return args


def predict_click():
    car_info = [int(e_year.get()), e_make.get(), e_model.get(),
                e_condition.get(), e_title.get(), e_transmission.get(), float(e_odometer.get())]
    car = car_info
    car_df = pd.DataFrame(data=[[car[0], car[1], car[2], car[3], car[4], car[5], car[6]]],
                          columns=['Year', 'Make', 'Model', 'Condition', 'Title', 'Transmission', 'Odometer'])
    car_df_dumb = match_dummies(car_df)
    car_value = float(model.predict(car_df_dumb))
    details_label = Label(root, text="Car details:", anchor=W).grid(row=8)
    year_label = Label(root, text=f"vehicle year: {car[0]}").grid(row=9)
    make_label = Label(root, text=f"vehicle make: {car[1]}").grid(row=10)
    model_label = Label(root, text=f"vehicle model: {car[2]}").grid(row=11)
    condition_label = Label(root, text=f"vehicle condition: {car[3]}").grid(row=12)
    title_label = Label(root, text=f"title status: {car[4]}").grid(row=13)
    transmission_label = Label(root, text=f"transmission type: {car[5]}").grid(row=14)
    odometer_label = Label(root, text=f"{car[6]} miles on odometer").grid(row=15)
    price_label = Label(root, text=f"Estimated price: ${car_value:,.2f}").grid(row=16)
    formatting_label = Label(root, text="").grid(row=17)


buttonPredict = Button(root, bg="gray", text="Predict Price", padx=50, pady=10, command=predict_click).grid(row=7)
buttonQuit = Button(root, bg='gray', text='Exit', padx=73, command=root.quit).grid(row=18)

root.mainloop()
