import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import json
import os
import argparse

#timeline display constants
NUM_HOUSES = 10
TIMELINE_PADDING = 40          # pixels on left for house labels
LANE_HEIGHT = 28          # pixels per house lane
TIMELINE_HEIGHT = NUM_HOUSES * LANE_HEIGHT + 20
TICK_INTERVAL = 96          # one day = 96 timesteps → one major tick

TYPE_COLORS = {
    "Fire":    "#e03030",
    "BBQ":     "#e07020",
    "Smoking": "#c8a800",
    "FDI_temp":"#7730c0",
    "FDI_pm":  "#c030a0",
    "Normal":  "#409040",
}

EDIT_FIELDS = [
    ("start_time", "Start Time",   int,   0),
    ("duration",   "Duration",     int,   1),
    ("severity",   "Severity",     float, 0.5),
]

class EventSchedulerApp:
    def __init__(self, root, output_path="scenario.json"):
        self.root = root
        self.output_path = output_path
        self.events = []
        self.selected_idx = None         # index into self.events

        self.root.title("IoT Event Scheduler")
        self.root.resizable(True, True)
        self._build_ui()

    def _build_ui(self):
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        #top bar scenario metadata + I/O
        top = ttk.Frame(self.root, padding=8)
        top.grid(row=0, column=0, sticky="ew")
        top.columnconfigure(3, weight=1)

        ttk.Label(top, text="Scenario Name:").grid(row=0, column=0, padx=(0, 4))
        self.var_name = tk.StringVar(value="scenario_1")
        ttk.Entry(top, textvariable=self.var_name, width=22).grid(row=0, column=1, padx=(0, 12))

        ttk.Label(top, text="Description:").grid(row=0, column=2, padx=(0, 4))
        self.var_desc = tk.StringVar(value="")
        ttk.Entry(top, textvariable=self.var_desc, width=40).grid(row=0, column=3, sticky="ew", padx=(0, 12))

        ttk.Button(top, text=" Load Events",  command=self._load_events).grid(row=0, column=4, padx=2)
        ttk.Button(top, text=" Save Scenario", command=self._save_scenario).grid(row=0, column=5, padx=2)

        #main pane list left + edit + timeline right
        main = ttk.PanedWindow(self.root, orient="horizontal")
        main.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 4))

        #left event list + inline editor
        left_frame = ttk.Frame(main, padding=4)
        main.add(left_frame, weight=1)
        left_frame.rowconfigure(0, weight=1)
        left_frame.columnconfigure(0, weight=1)

        self._build_list(left_frame)
        self._build_editor(left_frame)

        #right timeline canvas
        right_frame = ttk.LabelFrame(main, text=" Visual Timeline ", padding=4)
        main.add(right_frame, weight=3)
        right_frame.rowconfigure(0, weight=1)
        right_frame.columnconfigure(0, weight=1)

        self._build_timeline(right_frame)

        #bottom status bar
        self.status_var = tk.StringVar(value="Load an events.json file to begin.")
        ttk.Label(self.root, textvariable=self.status_var, relief="sunken", anchor="w").grid(row=2, column=0, sticky="ew", padx=8, pady=(0, 6))

    def _build_list(self, parent):
        list_frame = ttk.LabelFrame(parent, text=" Events ", padding=4)
        list_frame.grid(row=0, column=0, sticky="nsew")
        list_frame.rowconfigure(0, weight=1)
        list_frame.columnconfigure(0, weight=1)

        cols = ("id", "type", "house", "start", "dur", "sev")
        self.tree = ttk.Treeview(list_frame, columns=cols, show="headings", selectmode="browse", height=12)
        self.tree.heading("id", text="Event ID")
        self.tree.heading("type", text="Type")
        self.tree.heading("house", text="House")
        self.tree.heading("start", text="Start")
        self.tree.heading("dur", text="Dur.")
        self.tree.heading("sev", text="Sev.")

        self.tree.column("id", width=130, anchor="w")
        self.tree.column("type", width=85,  anchor="center")
        self.tree.column("house", width=45,  anchor="center")
        self.tree.column("start", width=55,  anchor="center")
        self.tree.column("dur", width=45,  anchor="center")
        self.tree.column("sev", width=45,  anchor="center")

        vsb = ttk.Scrollbar(list_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        self.tree.bind("<<TreeviewSelect>>", self._on_list_select)

        btn_row = ttk.Frame(list_frame)
        btn_row.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(4, 0))
        btn_row.columnconfigure((0, 1, 2), weight=1)
        ttk.Button(btn_row, text="↑↓ Sort by Start", command=self._sort_events).grid(row=0, column=0, sticky="ew", padx=2)
        ttk.Button(btn_row, text=" Remove", command=self._remove_event).grid(row=0, column=1, sticky="ew", padx=2)
        ttk.Button(btn_row, text=" Clear All", command=self._clear_all).grid(row=0, column=2, sticky="ew", padx=2)

    def _build_editor(self, parent):
        editor = ttk.LabelFrame(parent, text=" Edit Selected Event ", padding=6)
        editor.grid(row=1, column=0, sticky="ew", pady=(6, 0))
        editor.columnconfigure(1, weight=1)

        self._edit_vars = {}
        for r, (field, label, dtype, default) in enumerate(EDIT_FIELDS):
            ttk.Label(editor, text=f"{label}:").grid(row=r, column=0, sticky="w", pady=2)
            var = tk.StringVar(value=str(default))
            ttk.Entry(editor, textvariable=var, width=12).grid(row=r, column=1, sticky="w", pady=2)
            self._edit_vars[field] = (var, dtype)

        ttk.Button(editor, text=" Apply Changes", command=self._apply_edit).grid(row=len(EDIT_FIELDS), column=0, columnspan=2, sticky="ew", pady=(6, 2))
      
    def _build_timeline(self, parent):
        #scrollable canvas
        self.tl_canvas = tk.Canvas(parent, bg="#1e1e2e", height=TIMELINE_HEIGHT + 30, cursor="hand2")
        h_scroll = ttk.Scrollbar(parent, orient="horizontal", command=self.tl_canvas.xview)
        self.tl_canvas.configure(xscrollcommand=h_scroll.set)

        self.tl_canvas.grid(row=0, column=0, sticky="nsew")
        h_scroll.grid(row=1, column=0, sticky="ew")
        parent.rowconfigure(0, weight=1)

        self.tl_canvas.bind("<Configure>", lambda e: self._draw_timeline())
        self.tl_canvas.bind("<Button-1>", self._on_timeline_click)

    def _draw_timeline(self):
        c = self.tl_canvas
        c.delete("all")

        if not self.events:
            c.create_text(200, 60, text="No events loaded.", fill="#888", font=("monospace", 12))
            return

        max_time = max((ev['start_time'] + ev['duration']) for ev in self.events)
        display_time = max(max_time + 48, 288)     # at least 3 days visible
        canvas_w = max(c.winfo_width() - 20, 600)
        pixels_per_t = (canvas_w - TIMELINE_PADDING) / display_time

        #update scroll region
        total_w = int(display_time * pixels_per_t) + TIMELINE_PADDING + 20
        c.configure(scrollregion=(0, 0, total_w, TIMELINE_HEIGHT + 30))

        #house lane backgrounds
        for h in range(NUM_HOUSES):
            y = h * LANE_HEIGHT + 20
            bg = "#2a2a3e" if h % 2 == 0 else "#22223a"
            c.create_rectangle(0, y, total_w, y + LANE_HEIGHT, fill=bg, outline="")
            c.create_text(TIMELINE_PADDING - 4, y + LANE_HEIGHT // 2, text=f"H{h}", fill="#aaa", anchor="e", font=("monospace", 9))

        #day tick marks
        day = 0
        t   = 0
        while t <= display_time:
            x = TIMELINE_PADDING + t * pixels_per_t
            c.create_line(x, 18, x, TIMELINE_HEIGHT + 20, fill="#444", dash=(2, 4))
            c.create_text(x, 10, text=f"D{day+1}", fill="#666", font=("monospace", 8))
            t   += TICK_INTERVAL
            day += 1

        #event blocks
        for i, ev in enumerate(self.events):
            h = ev['target_house']
            start = ev['start_time']
            dur = ev['duration']
            etype = ev['event_type']
            sev = ev.get('severity', 1.0)

            x1 = TIMELINE_PADDING + start * pixels_per_t
            x2 = TIMELINE_PADDING + (start + dur) * pixels_per_t
            y1 = h * LANE_HEIGHT + 22
            y2 = y1 + LANE_HEIGHT - 4

            color = TYPE_COLORS.get(etype, "#888")
            outline = "#ffffff" if i == self.selected_idx else color
            lw  = 2  if i == self.selected_idx else 1
            alpha_color = color  # canvas doesn't support real alpha, use solid

            block = c.create_rectangle(x1, y1, x2, y2, fill=alpha_color, outline=outline, width=lw, tags=f"ev_{i}")
            label = etype[:3]
            if x2 - x1 > 18:
                c.create_text((x1 + x2) / 2, (y1 + y2) / 2, text=label, fill="white", font=("monospace", 8, "bold"), tags=f"ev_{i}")

            # Severity bar at bottom of block
            bar_h = max(2, int((y2 - y1 - 4) * sev))
            c.create_rectangle(x1 + 1, y2 - bar_h - 1, x1 + 3, y2 - 1, fill="white", outline="", tags=f"ev_{i}")

    def _on_timeline_click(self, event):
        #find which event block was clicked
        overlapping = self.tl_canvas.find_overlapping(event.x - 2, event.y - 2, event.x + 2, event.y + 2)
        for item in overlapping:
            tags = self.tl_canvas.gettags(item)
            for tag in tags:
                if tag.startswith("ev_"):
                    idx = int(tag[3:])
                    self._select_event(idx)
                    return

    def _on_list_select(self, _event=None):
        sel = self.tree.selection()
        if not sel:
            return
        ev_id = self.tree.item(sel[0])['values'][0]
        for i, ev in enumerate(self.events):
            if ev['event_id'] == ev_id:
                self._select_event(i)
                return

    def _select_event(self, idx):
        self.selected_idx = idx
        ev = self.events[idx]

        #highlight in tree
        for item in self.tree.get_children():
            if self.tree.item(item)['values'][0] == ev['event_id']:
                self.tree.selection_set(item)
                self.tree.see(item)
                break

        #populate editor
        for field, (var, dtype) in self._edit_vars.items():
            var.set(str(ev.get(field, '')))

        self._draw_timeline()
        self._status(f"Selected: {ev['event_id']}  ({ev['event_type']}  House {ev['target_house']})")

    def _apply_edit(self):
        if self.selected_idx is None:
            messagebox.showinfo("No Selection", "Select an event first.")
            return
        ev = self.events[self.selected_idx]
        for field, (var, dtype) in self._edit_vars.items():
            try:
                val = dtype(var.get())
            except ValueError:
                messagebox.showerror("Input Error", f"Invalid value for {field}.")
                return
            if field == 'severity' and not (0.0 <= val <= 1.0):
                messagebox.showerror("Input Error", "Severity must be 0.0–1.0.")
                return
            if field in ('start_time', 'duration') and val < 0:
                messagebox.showerror("Input Error", f"{field} must be ≥ 0.")
                return
            ev[field] = val
        self._refresh_list()
        self._draw_timeline()
        self._status(f"Updated {ev['event_id']}.")

    def _sort_events(self):
        self.events.sort(key=lambda e: (e['start_time'], e['target_house']))
        self.selected_idx = None
        self._refresh_list()
        self._draw_timeline()
        self._status("Events sorted by start time.")

    def _remove_event(self):
        if self.selected_idx is None:
            messagebox.showinfo("No Selection", "Select an event first.")
            return
        ev = self.events.pop(self.selected_idx)
        self.selected_idx = None
        self._refresh_list()
        self._draw_timeline()
        self._status(f"Removed {ev['event_id']}.")

    def _clear_all(self):
        if not self.events:
            return
        if messagebox.askyesno("Clear All", "Remove all events from the scenario?"):
            self.events.clear()
            self.selected_idx = None
            self._refresh_list()
            self._draw_timeline()
            self._status("All events cleared.")

    def _refresh_list(self):
        for item in self.tree.get_children():
            self.tree.delete(item)
        for ev in self.events:
            etype = ev['event_type']
            color = TYPE_COLORS.get(etype, "black")
            tag   = f"type_{etype}"
            self.tree.insert("", "end", values=(
                ev['event_id'],
                etype,
                ev['target_house'],
                ev['start_time'],
                ev['duration'],
                f"{ev.get('severity', 1.0):.2f}",
            ), tags=(tag,))
            self.tree.tag_configure(tag, foreground=color)
        count = len(self.events)
        self.root.title(f"IoT Event Scheduler — {count} event{'s' if count != 1 else ''}")

    def _load_events(self):
        path = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")])
        if not path:
            return
        try:
            with open(path) as f:
                data = json.load(f)
        except Exception as exc:
            messagebox.showerror("Load Error", f"Could not parse file:\n{exc}")
            return

        if isinstance(data, dict) and 'events' in data:
            loaded = data['events']
            if 'scenario_name' in data:
                self.var_name.set(data['scenario_name'])
            if 'description' in data:
                self.var_desc.set(data['description'])
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
        self._draw_timeline()
        self._status(f"Loaded {len(loaded)} event(s) from {os.path.basename(path)}")

    def _save_scenario(self):
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

        scenario = {
            "scenario_name": self.var_name.get().strip() or "scenario_1",
            "description": self.var_desc.get().strip(),
            "events": self.events,
        }
        with open(path, 'w') as f:
            json.dump(scenario, f, indent=2)
        self.output_path = path
        self._status(f"Saved scenario '{scenario['scenario_name']}' "
                     f"({len(self.events)} events) to {path}")

    def _status(self, msg):
        self.status_var.set(msg)

def main():
    parser = argparse.ArgumentParser(description="IoT Event Scheduler GUI")
    parser.add_argument("--load",   default=None, help="Events JSON file to pre-load")
    parser.add_argument("--output", default="scenario.json", help="Default save path (default: scenario.json)")
    args = parser.parse_args()
    root = tk.Tk()
    root.geometry("1100x620")
    app  = EventSchedulerApp(root, output_path=args.output)

    if args.load and os.path.exists(args.load):
        try:
            with open(args.load) as f:
                data = json.load(f)
            if isinstance(data, dict) and 'events' in data:
                app.events = data['events']
                if 'scenario_name' in data:
                    app.var_name.set(data['scenario_name'])
            elif isinstance(data, list):
                app.events = data
            app._refresh_list()
            app._draw_timeline()
            app._status(f"Pre-loaded {len(app.events)} events from {args.load}")
        except Exception as exc:
            print(f"Warning: could not pre-load {args.load}: {exc}")
    root.mainloop()

if __name__ == "__main__":
    main()
