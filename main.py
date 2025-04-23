import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
from scipy.stats import linregress
import os
from interpolation import lagrange_interpolation, newton_interpolation, cubic_spline, linear_regression

class PlantGrowthInterpolator:
    def __init__(self, root):
        self.root = root
        self.root.title("Analizador de Crecimiento de Plantas")
        self.root.geometry("1000x900")  # Slightly larger window
        self.root.configure(bg='#283618')  # Dark green background
        
        # Datos de ejemplo (tiempo en días, altura en cm)
        self.default_data = {
            "Basil": np.array([[0,1,4,5,6,7,8,9,11,12],
                               [7.22,7.5,8,12,15,18,24,37,42,54]]), #https://data.mendeley.com/datasets/vx4jy7wyvd/1/files/05bbf7c4-7bb6-445e-977e-fea44c9ab7b7
            "Tomate": np.array([[0, 1, 2, 3, 4, 5], 
                              [0, 1, 3, 6, 10, 15]]),
            "Rosa": np.array([[0, 2, 4, 6, 8, 10], 
                             [0, 3, 7, 10, 12, 13]]),
            "Girasol": np.array([[0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
                                [0, 5, 15, 30, 60, 100, 140, 170, 190, 200, 210]])  # Nueva planta añadida
        }
        
        self.current_data = None
        self.selected_plant = tk.StringVar()
        self.selected_method = tk.StringVar(value="Lagrange")
        self.time_input = tk.DoubleVar()
        self.custom_data = None
        
        self.create_widgets()
        self.load_plant_data("Basil")
        
    def create_widgets(self):
        self.style = ttk.Style()
        
        # Configure styles with green theme
        self.style.configure('.', background='#283618', foreground='#BC6C25')
        self.style.configure('TFrame', background='#283618')
        self.style.configure('TLabelFrame', background='##283618', 
                        foreground='#BC6C25', font=('urw gothic l', 10, 'bold'))
        self.style.configure('TLabel', background='#283618', 
                        foreground='#BC6C25')
        self.style.configure('TButton', background='#283618', foreground='#BC6C25',
                          font=('urw gothic l', 10, 'bold'))
        self.style.configure('TRadiobutton', background='#283618', foreground='#BC6C25',
                          font=('urw gothic l', 10, 'bold'))
        self.style.configure('TEntry', fieldbackground='#e8f5e9')
        self.style.map('TButton', 
                     background=[('active', '#DDA15E'), ('pressed', '#606C38')],
                     foreground=[('active','#FEFAE0'),('pressed','#DDA15E')])
        
        # Frame principal
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Título
        title_label = ttk.Label(main_frame, text="Analizador de Crecimiento de Plantas", 
                               font=("urw gothic l", 16, "bold"), foreground="#BC6C25")
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Imagen decorativa
        try:
            img = Image.open("plant_icon.png") if os.path.exists("plant_icon.png") else None
            if img:
                img = img.resize((250, 250), Image.LANCZOS)
                self.plant_img = ImageTk.PhotoImage(img)
                img_label = ttk.Label(main_frame, image=self.plant_img)
                img_label.grid(row=1, column=2, rowspan=4, padx=10)
        except Exception as e:
            print(f"Error loading image: {e}")
        
        # Selección de planta
        plant_frame = ttk.LabelFrame(main_frame, text="Seleccionar Planta", padding=10)
        plant_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=5)
        
        # Organizar los RadioButtons en 2 filas para mejor visualización
        plants = list(self.default_data.keys())
        half = len(plants) // 2 + len(plants) % 2
        
        for i, plant in enumerate(plants[:half]):
            rb = ttk.Radiobutton(plant_frame, text=plant, variable=self.selected_plant, 
                                value=plant, command=lambda: self.load_plant_data(self.selected_plant.get()))
            rb.grid(row=0, column=i, padx=5, pady=5)
        
        for i, plant in enumerate(plants[half:], start=half):
            rb = ttk.Radiobutton(plant_frame, text=plant, variable=self.selected_plant, 
                                value=plant, command=lambda: self.load_plant_data(self.selected_plant.get()))
            rb.grid(row=1, column=i-half, padx=5, pady=5)
        
        self.selected_plant.set("Basil")
        
        # Opción para datos personalizados
        custom_btn = ttk.Button(plant_frame, text="Ingresar Datos Personalizados", 
                              command=self.load_custom_data)
        custom_btn.grid(row=2, column=0, columnspan=len(plants), pady=5)
        
        # Método de interpolación
        method_frame = ttk.LabelFrame(main_frame, text="Método de Estimación", padding=10)
        method_frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=5)
        
        methods = ["Lagrange", "Newton", "Splines", "Regresión Lineal", "Regresión Exponencial"]
        for i, method in enumerate(methods):
            rb = ttk.Radiobutton(method_frame, text=method, variable=self.selected_method, 
                                value=method)
            rb.grid(row=0, column=i, padx=5, pady=5)
        
        # Tooltip for methods
        self.method_tooltips = {
            "Lagrange": "Polinomio que pasa exactamente por todos los puntos",
            "Newton": "Polinomio usando diferencias divididas (más eficiente que Lagrange)",
            "Splines": "Interpolación suave por segmentos cúbicos",
            "Regresión Lineal": "Ajuste lineal por mínimos cuadrados",
            "Regresión Exponencial": "Ajuste exponencial (y = a*e^(b*x))"
        }
        
        # Add tooltip functionality
        for method in methods:
            rb = ttk.Radiobutton(method_frame, text=method, variable=self.selected_method, 
                                value=method)
            rb.grid(row=0, column=i, padx=5, pady=5)
            rb.bind("<Enter>", lambda e, m=method: self.show_tooltip(m))
            rb.bind("<Leave>", lambda e: self.hide_tooltip())
        
        # Tooltip label
        self.tooltip_label = ttk.Label(method_frame, text="", background="#FEFAE0", 
                                     relief="solid", borderwidth=1, padding=5,
                                     font=('urw gothic l', 8))
        self.tooltip_label.grid(row=1, column=0, columnspan=len(methods), sticky="ew")
        self.tooltip_label.grid_remove()
        
        # Entrada de tiempo
        input_frame = ttk.LabelFrame(main_frame, text="Calcular Crecimiento en Tiempo", padding=10)
        input_frame.grid(row=3, column=0, columnspan=2, sticky="ew", pady=5)
        
        ttk.Label(input_frame, text="Tiempo (días):").grid(row=0, column=0, padx=5)
        time_entry = ttk.Entry(input_frame, textvariable=self.time_input, foreground="#BC6C25")
        time_entry.grid(row=0, column=1, padx=5)
        
        calc_btn = ttk.Button(input_frame, text="Calcular", command=self.calculate_growth)
        calc_btn.grid(row=0, column=2, padx=5)
        
        # Resultados
        self.result_label = ttk.Label(main_frame, text="Resultado: ", font=("urw gothic l", 12, 'bold'), 
                                    foreground="#BC6C25")
        self.result_label.grid(row=4, column=0, columnspan=2, pady=10)
        
        # Gráfico
        self.figure, self.ax = plt.subplots(figsize=(8, 5), facecolor='#e8f5e9')
        self.canvas = FigureCanvasTkAgg(self.figure, master=main_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.grid(row=5, column=0, columnspan=3, pady=20)
        
        # Actualizar gráfico inicial
        self.update_plot()
    
    def show_tooltip(self, method):
        self.tooltip_label.config(text=self.method_tooltips[method])
        self.tooltip_label.grid()
    
    def hide_tooltip(self):
        self.tooltip_label.grid_remove()
    
    def validate_data(self, data_str):
        """Validate and parse custom data input"""
        lines = data_str.strip().split('\n')
        if len(lines) < 2:
            raise ValueError("Se requieren al menos 2 puntos de datos")
        
        data = []
        for i, line in enumerate(lines):
            if not line.strip():
                continue
            parts = line.strip().split()
            if len(parts) != 2:
                raise ValueError(f"Línea {i+1}: Cada línea debe contener exactamente 2 números (tiempo y altura)")
            try:
                t, h = map(float, parts)
                if t < 0 or h < 0:
                    raise ValueError("Los valores no pueden ser negativos")
                data.append([t, h])
            except ValueError as e:
                raise ValueError(f"Línea {i+1}: Datos inválidos - {str(e)}")
        
        data = np.array(data).T
        if len(np.unique(data[0])) != len(data[0]):
            raise ValueError("Los valores de tiempo deben ser únicos")
        
        return data
    
    def load_plant_data(self, plant_name):
        """Load predefined plant data"""
        self.current_data = self.default_data[plant_name]
        self.custom_data = None
        self.update_plot()
        
    def load_custom_data(self):
        """Open window for custom data input"""
        custom_window = tk.Toplevel(self.root)
        custom_window.title("Ingresar Datos Personalizados")
        custom_window.geometry("500x400")
        custom_window.configure(bg='#283618')
        
        frame = ttk.Frame(custom_window, padding="20")
        frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(frame, text="Ingrese los datos (tiempo altura, uno por línea):", 
                font=("urw gothic l", 12)).pack(pady=10)
        
        # Text area for data input
        text_area = tk.Text(frame, height=15, width=50, bg='#e8f5e9')
        text_area.pack(pady=10, padx=20)
        text_area.insert(tk.END, "0 0\n1 2\n2 5\n3 9\n4 15\n5 22\n6 30\n7 40")
        
        def save_data():
            try:
                data = self.validate_data(text_area.get("1.0", tk.END))
                self.custom_data = data
                self.current_data = data
                self.selected_plant.set("Personalizado")
                self.update_plot()
                custom_window.destroy()
                messagebox.showinfo("Éxito", "Datos cargados correctamente")
            except Exception as e:
                messagebox.showerror("Error", f"Datos inválidos:\n{str(e)}")
        
        save_btn = ttk.Button(frame, text="Guardar Datos", command=save_data)
        save_btn.pack(pady=10)
        
    def calculate_growth(self):
        """Calculate plant growth at specified time using selected method"""
        if self.current_data is None:
            messagebox.showerror("Error", "No hay datos cargados")
            return
            
        t = self.time_input.get()
        time_data = self.current_data[0]
        growth_data = self.current_data[1]
        
        # Check for single data point
        if len(time_data) == 1:
            messagebox.showerror("Error", "Se requiere al menos 2 puntos de datos para interpolación")
            return
        
        # Check for extrapolation
        extrapolating = t < min(time_data) or t > max(time_data)
        if extrapolating and not messagebox.askyesno("Advertencia", 
                                     "El tiempo está fuera del rango de datos. ¿Desea extrapolar?"):
            return
        
        try:
            method = self.selected_method.get()
            height = 0
            
            if method == "Lagrange":
                height = lagrange_interpolation(time_data, growth_data, t)
            elif method == "Newton":
                height = newton_interpolation(time_data, growth_data, t)
            elif method == "Splines":
                height = cubic_spline(time_data, growth_data, t)
            elif method == "Regresión Lineal":
                (slope, intercept), predict = linear_regression(time_data, growth_data)
                height = predict(t)
            elif method == "Regresión Exponencial":
                log_y = np.log(growth_data)
                slope, intercept, _, _, _ = linregress(time_data, log_y)
                height = np.exp(intercept + slope * t)
            
            self.result_label.config(text=f"Resultado: En el día {t:.1f}, la planta tendrá una altura de {height:.2f} cm")
            
            # Update plot with the calculated point
            self.update_plot(highlight_point=(t, height))
            
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo calcular:\n{str(e)}")
    
    def update_plot(self, highlight_point=None):
        """Update the growth plot with current data and method"""
        self.ax.clear()
        
        if self.current_data is not None:
            time_data = self.current_data[0]
            growth_data = self.current_data[1]
            
            # Plot data points
            self.ax.scatter(time_data, growth_data, color='#2e7d32', s=100, 
                           label='Datos observados', zorder=3)
            
            # Plot interpolation/regression
            if len(time_data) > 1:
                fine_time = np.linspace(min(time_data), max(time_data), 200)
                
                method = self.selected_method.get()
                if method == "Lagrange":
                    y_values = [lagrange_interpolation(time_data, growth_data, x) for x in fine_time]
                    self.ax.plot(fine_time, y_values, '--', color='#7cb342', 
                                label='Interpolación de Lagrange', linewidth=2)
                elif method == "Newton":
                    y_values = [newton_interpolation(time_data, growth_data, x) for x in fine_time]
                    self.ax.plot(fine_time, y_values, '--', color='#7cb342', 
                                label='Interpolación de Newton', linewidth=2)
                elif method == "Splines":
                    y_values = cubic_spline(time_data, growth_data, fine_time)
                    self.ax.plot(fine_time, y_values, '--', color='#7cb342', 
                                label='Interpolación por splines cúbicos', linewidth=2)
                elif method == "Regresión Lineal":
                    (slope, intercept), _ = linear_regression(time_data, growth_data)
                    self.ax.plot(fine_time, slope*fine_time + intercept, '--', color='#7cb342', 
                                label='Regresión lineal', linewidth=2)
                elif method == "Regresión Exponencial":
                    log_y = np.log(growth_data)
                    slope, intercept, _, _, _ = linregress(time_data, log_y)
                    self.ax.plot(fine_time, np.exp(intercept + slope*fine_time), '--', 
                                color='#7cb342', label='Regresión exponencial', linewidth=2)
            
            # Highlight calculated point
            if highlight_point:
                t, h = highlight_point
                self.ax.scatter([t], [h], color='#d32f2f', s=150, 
                              label='Estimación actual', zorder=5)
                self.ax.annotate(f'{h:.1f} cm', (t, h), textcoords="offset points", 
                               xytext=(10,10), ha='center', color='#d32f2f', 
                               fontsize=10, bbox=dict(boxstyle='round,pad=0.5', 
                               fc='white', alpha=0.7))
        
        # Configure plot appearance
        self.ax.set_facecolor('#e8f5e9')
        title = f"Crecimiento de la Planta: {self.selected_plant.get()}"
        if self.selected_method.get() in ["Regresión Lineal", "Regresión Exponencial"]:
            title += f" ({self.selected_method.get()})"
        self.ax.set_title(title, pad=20, fontsize=12)
        self.ax.set_xlabel('Tiempo (días)', color='#2e7d32', fontsize=10)
        self.ax.set_ylabel('Altura (cm)', color='#2e7d32', fontsize=10)
        self.ax.grid(True, linestyle='--', alpha=0.7)
        self.ax.legend(facecolor='#e8f5e9', framealpha=0.8)
        
        self.figure.tight_layout()
        self.canvas.draw()
        
if __name__ == "__main__":
    root = tk.Tk()
    app = PlantGrowthInterpolator(root)
    root.mainloop()