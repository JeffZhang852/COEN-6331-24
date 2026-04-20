import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import json
import uuid
import argparse
import os

#supported event types and which extra field each needs
EVENT_TYPES = ["Fire", "BBQ", "Smoking", "FDI_temp", "FDI_pm"]
FDI_EVENT_TYPES = {"FDI_temp", "FDI_pm"}
FDI_SENSORS = {
    "FDI_temp": ["outdoor_temp", "indoor_temp"],
    "FDI_pm":   ["outdoor_pm",  "indoor_pm"],
}

DEFAULT_DURATIONS = {"Fire": 8, "BBQ": 4, "Smoking": 2, "FDI_temp": 6, "FDI_pm": 6}
DEFAULT_SEVERITY  = {"Fire": 0.9, "BBQ": 0.7, "Smoking": 0.5, "FDI_temp": 0.7, "FDI_pm": 0.7}

#colours for the event type badges in the list
TYPE_COLORS = {
    "Fire":    "#ff4444",
    "BBQ":     "#ff8800",
    "Smoking": "#ccaa00",
    "FDI_temp":"#8844cc",
    "FDI_pm":  "#cc44aa",
}

class EventGeneratorApp:
    def __init__(self, root, output_path="events.json"):
        self.root        = root
        self.output_path = output_path
        self.events      = []           # list of event dicts
        self.selected_idx = None        # index currently selected in the list

        self.root.title("IoT Event Generator")
        self.root.resizable(True, True)
        self._build_ui()
        self._refresh_list()

    def _build_ui(self):
        self.root.columnconfigure(0, weight=1)
        self.root.columnconfigure(1, weight=2)
        self.root.rowconfigure(0, weight=1)

        #left panel form
        form_frame = ttk.LabelFrame(self.root, text=" Event Properties ", padding=10)
        form_frame.grid(row=0, column=0, sticky="nsew", padx=(10, 5), pady=10)
        form_frame.columnconfigure(1, weight=1)
        row = 0

        #event type
        ttk.Label(form_frame, text="Event Type:").grid(row=row, column=0, sticky="w", pady=3)
        self.var_type = tk.StringVar(value=EVENT_TYPES[0])
        self.cb_type  = ttk.Combobox(form_frame, textvariable=self.var_type,
                                     values=EVENT_TYPES, state="readonly", width=14)
        self.cb_type.grid(row=row, column=1, sticky="ew", pady=3)
        self.cb_type.bind("<<ComboboxSelected>>", self._on_type_change)
        row += 1

        #target house
        ttk.Label(form_frame, text="Target House (0–9):").grid(row=row, column=0, sticky="w", pady=3)
        self.var_house = tk.IntVar(value=0)
        sb_house = ttk.Spinbox(form_frame, from_=0, to=9, textvariable=self.var_house, width=6)
        sb_house.grid(row=row, column=1, sticky="w", pady=3)
        row += 1

        #start time
        ttk.Label(form_frame, text="Start Time (timestep):").grid(row=row, column=0, sticky="w", pady=3)
        self.var_start = tk.IntVar(value=0)
        ttk.Entry(form_frame, textvariable=self.var_start, width=10).grid(row=row, column=1, sticky="w", pady=3)
        row += 1

        #duration
        ttk.Label(form_frame, text="Duration (timesteps):").grid(row=row, column=0, sticky="w", pady=3)
        self.var_duration = tk.IntVar(value=DEFAULT_DURATIONS[EVENT_TYPES[0]])
        ttk.Entry(form_frame, textvariable=self.var_duration, width=10).grid(row=row, column=1, sticky="w", pady=3)
        row += 1

        #severity
        ttk.Label(form_frame, text="Severity (0.0 – 1.0):").grid(row=row, column=0, sticky="w", pady=3)
        self.var_severity = tk.DoubleVar(value=DEFAULT_SEVERITY[EVENT_TYPES[0]])
        sev_entry = ttk.Entry(form_frame, textvariable=self.var_severity, width=10)
        sev_entry.grid(row=row, column=1, sticky="w", pady=3)
        row += 1

        #severity slider (synced to entry)
        self.sev_slider = ttk.Scale(form_frame, from_=0.0, to=1.0, orient="horizontal", command=self._on_slider_move)
        self.sev_slider.set(self.var_severity.get())
        self.sev_slider.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(0, 6))
        row += 1

        #sensor type (FDI only)
        self.lbl_sensor = ttk.Label(form_frame, text="Sensor Target:")
        self.lbl_sensor.grid(row=row, column=0, sticky="w", pady=3)
        self.var_sensor  = tk.StringVar(value=FDI_SENSORS["FDI_temp"][0])
        self.cb_sensor   = ttk.Combobox(form_frame, textvariable=self.var_sensor, values=FDI_SENSORS["FDI_temp"], state="readonly", width=14)
        self.cb_sensor.grid(row=row, column=1, sticky="ew", pady=3)
        row += 1

        #fake value - FDI only
        self.lbl_fakeval = ttk.Label(form_frame, text="Fake Value (opt.):")
        self.lbl_fakeval.grid(row=row, column=0, sticky="w", pady=3)
        self.var_fakeval = tk.StringVar(value="")
        self.ent_fakeval = ttk.Entry(form_frame, textvariable=self.var_fakeval, width=10)
        self.ent_fakeval.grid(row=row, column=1, sticky="w", pady=3)
        row += 1

        ttk.Separator(form_frame, orient="horizontal").grid(row=row, column=0, columnspan=2, sticky="ew", pady=8)
        row += 1

        #buttons
        btn_frame = ttk.Frame(form_frame)
        btn_frame.grid(row=row, column=0, columnspan=2, sticky="ew")
        btn_frame.columnconfigure((0, 1), weight=1)

        ttk.Button(btn_frame, text="  Add Event", command=self._add_event).grid(row=0, column=0, sticky="ew", padx=2, pady=2)
        ttk.Button(btn_frame, text="  Update Selected", command=self._update_event).grid(row=0, column=1, sticky="ew", padx=2, pady=2)
        ttk.Button(btn_frame, text="  Remove Selected", command=self._remove_event).grid(row=1, column=0, sticky="ew", padx=2, pady=2)
        ttk.Button(btn_frame, text="  Clear All", command=self._clear_all).grid(row=1, column=1, sticky="ew", padx=2, pady=2)
        row += 1

        ttk.Separator(form_frame, orient="horizontal").grid(row=row, column=0, columnspan=2, sticky="ew", pady=8)
        row += 1
        io_frame = ttk.Frame(form_frame)
        io_frame.grid(row=row, column=0, columnspan=2, sticky="ew")
        io_frame.columnconfigure((0, 1), weight=1)

        ttk.Button(io_frame, text=" Save JSON", command=self._save_json).grid(row=0, column=0, sticky="ew", padx=2, pady=2)
        ttk.Button(io_frame, text=" Load JSON", command=self._load_json).grid(row=0, column=1, sticky="ew", padx=2, pady=2)

        #initially hide FDI specific fields if not in FDI mode
        self._on_type_change()

        #right panel event list
        list_frame = ttk.LabelFrame(self.root, text=" Event List ", padding=10)
        list_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 10), pady=10)
        list_frame.rowconfigure(0, weight=1)
        list_frame.columnconfigure(0, weight=1)

        columns = ("id", "type", "house", "start", "dur", "sev")
        self.tree = ttk.Treeview(list_frame, columns=columns, show="headings", selectmode="browse")
        self.tree.heading("id",    text="Event ID")
        self.tree.heading("type",  text="Type")
        self.tree.heading("house", text="House")
        self.tree.heading("start", text="Start")
        self.tree.heading("dur",   text="Duration")
        self.tree.heading("sev",   text="Severity")

        self.tree.column("id",    width=140, anchor="w")
        self.tree.column("type",  width=90,  anchor="center")
        self.tree.column("house", width=55,  anchor="center")
        self.tree.column("start", width=60,  anchor="center")
        self.tree.column("dur",   width=65,  anchor="center")
        self.tree.column("sev",   width=65,  anchor="center")

        vsb = ttk.Scrollbar(list_frame, orient="vertical",   command=self.tree.yview)
        hsb = ttk.Scrollbar(list_frame, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        self.tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        
        self.tree.bind("<<TreeviewSelect>>", self._on_select)

        #status bar
        self.status_var = tk.StringVar(value="Ready. Add events using the form.")
        ttk.Label(self.root, textvariable=self.status_var, relief="sunken", anchor="w").grid(row=1, column=0, columnspan=2, sticky="ew", padx=10, pady=(0, 6))
      
    def _on_type_change(self, _event=None):
        etype = self.var_type.get()
        is_fdi = etype in FDI_EVENT_TYPES

        #show-hide FDI only rows
        state = "normal" if is_fdi else "disabled"
        self.lbl_sensor.configure(state=state)
        self.cb_sensor.configure(state="readonly" if is_fdi else "disabled")
        self.lbl_fakeval.configure(state=state)
        self.ent_fakeval.configure(state=state)

        if is_fdi:
            sensors = FDI_SENSORS.get(etype, [])
            self.cb_sensor.configure(values=sensors)
            self.var_sensor.set(sensors[0] if sensors else "")

        #auto fill sensible defaults
        self.var_duration.set(DEFAULT_DURATIONS.get(etype, 4))
        self.var_severity.set(DEFAULT_SEVERITY.get(etype, 0.7))
        self.sev_slider.set(self.var_severity.get())

    def _on_slider_move(self, val):
        self.var_severity.set(round(float(val), 2))

    def _on_select(self, _event=None):
        sel = self.tree.selection()
        if not sel:
            self.selected_idx = None
            return
        item  = self.tree.item(sel[0])
        ev_id = item['values'][0]

        for i, ev in enumerate(self.events):
            if ev['event_id'] == ev_id:
                self.selected_idx = i
                self._populate_form(ev)
                return

    def _populate_form(self, ev):
        #fill form fields from an event dict
        self.var_type.set(ev['event_type'])
        self._on_type_change()
        self.var_house.set(ev['target_house'])
        self.var_start.set(ev['start_time'])
        self.var_duration.set(ev['duration'])
        self.var_severity.set(ev['severity'])
        self.sev_slider.set(ev['severity'])
        params = ev.get('parameters', {})
        self.var_sensor.set(params.get('sensor', ''))
        fv = params.get('fake_val', '')
        self.var_fakeval.set(str(fv) if fv != '' else '')

    def _collect_form(self):
        #validate and collect form data - returns event dict or None"""
        etype = self.var_type.get()

        try:
            house    = int(self.var_house.get())
            start    = int(self.var_start.get())
            duration = int(self.var_duration.get())
            severity = float(self.var_severity.get())
        except (ValueError, tk.TclError) as exc:
            messagebox.showerror("Input Error", f"Invalid numeric value:\n{exc}")
            return None

        if not (0 <= house <= 9):
            messagebox.showerror("Input Error", "House must be 0–9.")
            return None
        if start < 0:
            messagebox.showerror("Input Error", "Start time must be ≥ 0.")
            return None
        if duration < 1:
            messagebox.showerror("Input Error", "Duration must be ≥ 1.")
            return None
        if not (0.0 <= severity <= 1.0):
            messagebox.showerror("Input Error", "Severity must be between 0.0 and 1.0.")
            return None

        params = {}
        if etype in FDI_EVENT_TYPES:
            params['sensor'] = self.var_sensor.get()
            fv_str = self.var_fakeval.get().strip()
            if fv_str:
                try:
                    params['fake_val'] = float(fv_str)
                except ValueError:
                    messagebox.showerror("Input Error", "Fake Value must be a number.")
                    return None
        return {
            "event_type":   etype,
            "target_house": house,
            "start_time":   start,
            "duration":     duration,
            "severity":     round(severity, 4),
            "parameters":   params,
        }

    def _add_event(self):
        data = self._collect_form()
        if data is None:
            return
        data['event_id'] = f"evt_{uuid.uuid4().hex[:8]}"
        self.events.append(data)
        self._refresh_list()
        self._status(f"Added {data['event_type']} event for House {data['target_house']}.")

    def _update_event(self):
        if self.selected_idx is None:
            messagebox.showinfo("No Selection", "Select an event in the list first.")
            return
        data = self._collect_form()
        if data is None:
            return
        data['event_id'] = self.events[self.selected_idx]['event_id']
        self.events[self.selected_idx] = data
        self._refresh_list()
        self._status(f"Updated event {data['event_id']}.")

    def _remove_event(self):
        if self.selected_idx is None:
            messagebox.showinfo("No Selection", "Select an event in the list first.")
            return
        ev = self.events.pop(self.selected_idx)
        self.selected_idx = None
        self._refresh_list()
        self._status(f"Removed event {ev['event_id']}.")

    def _clear_all(self):
        if not self.events:
            return
        if messagebox.askyesno("Clear All", "Remove all events?"):
            self.events.clear()
            self.selected_idx = None
            self._refresh_list()
            self._status("All events cleared.")

    def _refresh_list(self):
        for item in self.tree.get_children():
            self.tree.delete(item)

        for ev in self.events:
            etype  = ev['event_type']
            color  = TYPE_COLORS.get(etype, "black")
            tag    = f"type_{etype}"
            self.tree.insert("", "end", values=(
                ev['event_id'],
                etype,
                ev['target_house'],
                ev['start_time'],
                ev['duration'],
                f"{ev['severity']:.2f}",
            ), tags=(tag,))
            self.tree.tag_configure(tag, foreground=color)

        count = len(self.events)
        self.root.title(f"IoT Event Generator — {count} event{'s' if count != 1 else ''}")

    def _save_json(self):
        if not self.events:
            messagebox.showwarning("Empty", "No events to save.")
            return
        path = filedialog.asksaveasfilename(
            initialfile=os.path.basename(self.output_path),
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return
        payload = {"events": self.events}
        with open(path, 'w') as f:
            json.dump(payload, f, indent=2)
        self.output_path = path
        self._status(f"Saved {len(self.events)} event(s) to {path}")

    def _load_json(self):
        path = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if not path:
            return
        try:
            with open(path) as f:
                data = json.load(f)
        except Exception as exc:
            messagebox.showerror("Load Error", f"Could not parse JSON:\n{exc}")
            return

        if isinstance(data, dict) and 'events' in data:
            loaded = data['events']
        elif isinstance(data, list):
            loaded = data
        else:
            messagebox.showerror("Load Error", "Unrecognised JSON structure.")
            return

        if self.events and not messagebox.askyesno("Replace?", f"Replace current {len(self.events)} event(s) with {len(loaded)} loaded?"):
            return

        self.events = loaded
        self.selected_idx = None
        self._refresh_list()
        self._status(f"Loaded {len(loaded)} event(s) from {path}")

    def _status(self, msg):
        self.status_var.set(msg)

def main():
    parser = argparse.ArgumentParser(description="IoT Event Generator GUI")
    parser.add_argument("--output", default="events.json", help="Default save path (default: events.json)")
    args = parser.parse_args()

    root = tk.Tk()
    root.geometry("900x540")
    app  = EventGeneratorApp(root, output_path=args.output)
    root.mainloop()


if __name__ == "__main__":
    main()
