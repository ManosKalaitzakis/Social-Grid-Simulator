# -*- coding: utf-8 -*-
"""
Social Grid Simulator — Professional Dark UI (1920×1080)
========================================================
Python 3.10+. Single file. Requires pygame:
    python -m pip install pygame

Key features
------------
- Professional dark theme, 1920×1080 layout.
- Help panel height computed from text (no overlap).
- History grid 2×5 in Selected panel (last 10).
- Women die when beauty <= 0.05 (checked hourly & at end of day).
- Live knobs: women accept top %, men beauty threshold.
- Logging to CSV + events log; save/load; leaderboards; charts.
- NEW: distinct idle/interaction icons for men & women.
- NEW: larger fonts (+10%), orange headers, improved leaderboard.
- NEW: charts with axis labels; births/deaths split by gender.

Hotkeys
-------
Space pause/resume, R reset, +/- speed
G grid, H help, F fast forward
S/L save/load, O leaderboard overlay
Click agent; N/B next/prev
[ / ] women accept top %,  ; / ' men beauty threshold
J snapshot, K toggle hourly logging
TAB or 1/2/3 switch panel page
Wheel or Up/Down scroll right panel
"""

from __future__ import annotations
import os, sys, json, math, random, time, csv
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional

# ----- Windows-safe UTF-8 -----
if sys.stdout and sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    except Exception:
        pass

try:
    import pygame
except Exception as e:
    print("Install pygame first:  python -m pip install pygame")
    print(e); sys.exit(1)

# =====================================================================================
# CONFIG — all experiment knobs
# =====================================================================================

@dataclass
class Config:
    # Display / Grid (fits 1920×1080 exactly)
    screen_w: int = 1920
    screen_h: int = 1080
    cell: int = 20
    cols: int = 70            # 70*20 + 520 = 1920
    rows: int = 54            # 54*20 = 1080
    ui_panel_w: int = 520
    caption: str = "Social Grid — Professional Dark"
    fps_limit: int = 60

    # World geometry
    left_margin: int = 2
    right_margin: int = 2
    men_block_w: int = 12
    women_block_w: int = 12
    center_w: int = 14

    # Population
    n_men: int = 48
    n_women: int = 48
    seed: int = 2025

    # Time mapping
    ticks_per_hour: int = 60   # 60 ticks ≈ 1 hour
    start_hour: int = 7
    hours_per_day: int = 24

    # Movement / interaction
    interaction_radius: int = 2
    interaction_lock_ticks: int = 25
    cafe_session_cap_min: int = 240
    cafe_idle_cap_min: int = 120

    # Acceptance knobs (live adjustable)
    top_percent_default: float = 0.10           # women accept top-X% of men
    men_beauty_threshold_default: float = 0.00  # “0 = no standards”

    # Attribute dynamics
    conf_gain_on_accept: float = 0.03
    conf_loss_on_reject: float = 0.02
    conf_daily_drift: float = -0.005
    conf_bounds: Tuple[float, float] = (0.05, 1.0)

    beauty_decay_no_match_per_day: float = 0.06
    beauty_recovery_on_match: float = 0.02
    beauty_bounds: Tuple[float, float] = (0.05, 1.0)  # min is lethal now

    # Daily home quota beyond sleep
    min_home_minutes_per_day: int = 60

    # Isolation & death
    isolation_conf_threshold: float = 0.05
    isolation_death_days: int = 2

    # Birth control
    accepts_per_child: int = 5
    birth_cooldown_days: int = 3
    base_birth_prob: float = 1.00
    pressure_k: float = 1.35
    target_population_total: int = 96
    soft_cap_total: int = 180

    # Natural mortality
    natural_mortality_age_days: int = 120
    natural_mortality_base: float = 0.002
    natural_mortality_age_k: float = 0.015
    natural_mortality_low_attr_bonus: float = 0.010

    # Logging
    log_csv_path: str = "sim_log.csv"
    log_events_path: str = "events.log"
    log_every_hour: bool = True
    log_rotate_max_kb: int = 8 * 1024
    save_path: str = "sim_save.json"

    # UI toggles
    draw_grid: bool = False
    show_help: bool = True
    fast_forward: bool = False
    tpf: int = 1
    show_leaderboard: bool = False
    panel_page: int = 1
    panel_scroll_speed: int = 48
    panel_scrollpad: int = 8

    # Derived
    width: int = field(init=False)
    height: int = field(init=False)
    def __post_init__(self):
        self.width = self.screen_w
        self.height = self.screen_h
        assert self.cols * self.cell + self.ui_panel_w == self.screen_w
        assert self.rows * self.cell == self.screen_h

# =====================================================================================

class Colors:
    # Professional dark palette
    BG0=(18,20,24)    # page background
    BG1=(24,26,32)    # tiles A
    BG2=(28,30,38)    # tiles B
    BG3=(34,36,44)    # panels
    BG4=(42,44,54)    # panel header
    BG5=(12,13,16)    # shadows

    FG0=(236,239,244) # strong text
    FG1=(200,205,214) # normal text
    FG2=(150,155,165) # muted text

    ACC1=(90,160,255)   # men primary
    ACC2=(245,160,95)   # women primary (orange)
    ACC3=(110,190,140)  # ok
    ACC4=(230,110,110)  # warn
    ACC5=(255,220,120)  # header orange tint

    # Buildings
    HOME_M=(52,86,150)
    HOME_W=(150,92,52)
    WORK=(62,110,85)
    CAFE=(105,80,120)

# =====================================================================================
# Helpers
# =====================================================================================

def clamp(v, lo, hi): return lo if v<lo else hi if v>hi else v
def manhattan(a:Tuple[int,int], b:Tuple[int,int])->int: return abs(a[0]-b[0])+abs(a[1]-b[1])
def within(c:Config, p:Tuple[int,int])->bool: return 0<=p[0]<c.cols and 0<=p[1]<c.rows
def grid_to_px(c:Config, p:Tuple[int,int])->Tuple[int,int]: return p[0]*c.cell+c.cell//2, p[1]*c.cell+c.cell//2
def rect_for_cell(c:Config, p:Tuple[int,int])->pygame.Rect: return pygame.Rect(p[0]*c.cell, p[1]*c.cell, c.cell, c.cell)

def ensure_csv_header(path: str, header: List[str]):
    if not os.path.exists(path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(header)

def rotate_if_big(path: str, max_kb: int):
    try:
        if os.path.exists(path) and os.path.getsize(path) > max_kb * 1024:
            base, ext = os.path.splitext(path)
            ts = time.strftime("%Y%m%d_%H%M%S")
            os.replace(path, f"{base}_{ts}{ext}")
    except Exception:
        pass

# =====================================================================================
# World layout and labeled buildings
# =====================================================================================

@dataclass
class Building:
    name: str
    color: Tuple[int,int,int]
    cells: List[Tuple[int,int]]

class World:
    def __init__(self, cfg: Config, rng: random.Random):
        self.c=cfg; self.rng=rng
        self.men_homes: List[Building]=[]
        self.women_homes: List[Building]=[]
        self.work: Building
        self.cafe: Building
        self._build()

    def _block(self, x0:int, x1:int, y0:int, y1:int)->List[Tuple[int,int]]:
        return [(x,y) for x in range(x0, x1+1) for y in range(y0, y1+1)]

    def _build(self):
        band_h = self.c.rows // 6
        gap = 1

        # Men homes left
        lx0 = self.c.left_margin
        lx1 = lx0 + self.c.men_block_w - 1
        for i in range(5):
            y0 = gap + i*(band_h+gap)
            y1 = min(self.c.rows-2, y0 + band_h - 1)
            self.men_homes.append(Building(f"MenHome{i+1}", Colors.HOME_M, self._block(lx0, lx1, y0, y1)))

        # Women homes right
        rx1 = self.c.cols - self.c.right_margin - 1
        rx0 = rx1 - self.c.women_block_w + 1
        for i in range(5):
            y0 = gap + i*(band_h+gap)
            y1 = min(self.c.rows-2, y0 + band_h - 1)
            self.women_homes.append(Building(f"WomenHome{i+1}", Colors.HOME_W, self._block(rx0, rx1, y0, y1)))

        # Work / Cafe center
        cx0 = (self.c.cols - self.c.center_w)//2
        cx1 = cx0 + self.c.center_w - 1
        upper_h = self.c.rows//2 - 3
        self.work = Building("Work", Colors.WORK, self._block(cx0, cx1, 3, upper_h))
        self.cafe = Building("Cafe", Colors.CAFE, self._block(cx0, cx1, upper_h+3, self.c.rows-4))

    def random_cell(self, b: Building) -> Tuple[int,int]:
        return self.rng.choice(b.cells)

# =====================================================================================
# People
# =====================================================================================

@dataclass
class Man:
    id: int
    pos: Tuple[int,int]
    home: Building
    shift: str
    money: float
    power: float
    health: float
    confidence: float
    target: Tuple[int,int]
    lock_ticks: int = 0
    last_match_day: int = -9999
    last_result: str = ""
    home_minutes_today: int = 0
    age_days: int = 0
    cafe_session_min: int = 0
    cafe_idle_min: int = 0
    minutes_since_any_interaction: int = 0
    accepts_total: int = 0
    rejects_total: int = 0
    history: List[Tuple[int,int,str]] = field(default_factory=list)
    isolated: bool = False
    isolation_start_day: int = -9999
    dead: bool = False
    def score(self)->float:
        return clamp(0.25*self.money + 0.30*self.power + 0.25*self.health + 0.20*self.confidence, 0.0, 1.0)

@dataclass
class Woman:
    id: int
    pos: Tuple[int,int]
    home: Building
    shift: str
    beauty: float
    target: Tuple[int,int]
    lock_ticks: int = 0
    last_match_day: int = -9999
    last_result: str = ""
    home_minutes_today: int = 0
    age_days: int = 0
    cafe_session_min: int = 0
    cafe_idle_min: int = 0
    minutes_since_any_interaction: int = 0
    accepts_total: int = 0
    rejects_total: int = 0
    births: int = 0
    history: List[Tuple[int,int,str]] = field(default_factory=list)
    isolated: bool = False
    isolation_start_day: int = -9999
    dead: bool = False
    def sociality(self)->float:
        return clamp(0.4 + 0.6*self.beauty, 0.0, 1.0)

# =====================================================================================
# Population + rules
# =====================================================================================

class Population:
    def __init__(self, cfg: Config, world: World, rng: random.Random):
        self.c=cfg; self.world=world; self.rng=rng
        self.men: List[Man]=[]; self.women: List[Woman]=[]
        self.day_count=0; self.hour=cfg.start_hour%cfg.hours_per_day; self.tick_in_hour=0

        # runtime knobs
        self.women_accept_top_percent = cfg.top_percent_default
        self.men_beauty_threshold = cfg.men_beauty_threshold_default

        # daily tallies
        self.matches_today=0; self.rejects_today=0
        self.births_today=0; self.deaths_today=0
        self.births_today_m=0; self.births_today_w=0
        self.deaths_today_m=0; self.deaths_today_w=0

        # totals
        self.total_matches=0; self.total_rejects=0
        self.total_births=0; self.total_deaths=0
        self.total_births_m=0; self.total_births_w=0
        self.total_deaths_m=0; self.total_deaths_w=0

        # history for charts (60 days)
        self.days_hist: List[int]=[]
        self.hist_pop: List[int]=[]
        self.hist_births: List[int]=[]
        self.hist_deaths: List[int]=[]
        self.hist_births_m: List[int]=[]
        self.hist_births_w: List[int]=[]
        self.hist_deaths_m: List[int]=[]
        self.hist_deaths_w: List[int]=[]
        self.hist_matches: List[int]=[]
        self.hist_rejects: List[int]=[]

        # logging setup
        self._csv_header = [
            "ts","day","hour","minute","men","women","alive_total",
            "w_accept_top_pct","m_beauty_thr","threshold_score",
            "matches_today","rejects_today",
            "births_today","births_today_m","births_today_w",
            "deaths_today","deaths_today_m","deaths_today_w",
            "total_matches","total_rejects",
            "total_births","total_births_m","total_births_w",
            "total_deaths","total_deaths_m","total_deaths_w",
            "avg_score_men","avg_conf_men","avg_beauty_women",
            "at_home","at_work","at_cafe"
        ]
        rotate_if_big(self.c.log_csv_path, self.c.log_rotate_max_kb)
        ensure_csv_header(self.c.log_csv_path, self._csv_header)
        rotate_if_big(self.c.log_events_path, self.c.log_rotate_max_kb)

        self._init_people()
        self._update_top_threshold()
        self._log_snapshot()

    def _assign_shift(self, i:int)->str:
        return "MORNING" if (i + self.rng.randint(0,1)) % 2 == 0 else "AFTERNOON"

    def _init_people(self):
        for i in range(self.c.n_men):
            home = self.world.men_homes[i % len(self.world.men_homes)]
            pos = self.world.random_cell(home)
            m = Man(
                id=i+1, pos=pos, home=home, shift=self._assign_shift(i), target=pos,
                money=self.rng.uniform(0.2,0.9), power=self.rng.uniform(0.2,0.9),
                health=self.rng.uniform(0.3,0.95), confidence=self.rng.uniform(0.25,0.85)
            )
            self.men.append(m)
        for i in range(self.c.n_women):
            home = self.world.women_homes[i % len(self.world.women_homes)]
            pos = self.world.random_cell(home)
            w = Woman(
                id=i+1, pos=pos, home=home, shift=self._assign_shift(i), target=pos,
                beauty=self.rng.uniform(0.3,0.95)
            )
            self.women.append(w)

    # ---- Time helpers ----
    def _minutes_now(self)->int:
        minute = int(self.tick_in_hour * 60 / self.c.ticks_per_hour)
        return (self.hour*60 + minute) % 1440
    def _hhmm(self)->int:
        m=self._minutes_now(); return (m//60)*100 + (m%60)

    def advance_tick(self):
        self.tick_in_hour += 1
        if self.tick_in_hour % max(1, self.c.ticks_per_hour // 60) == 0:
            self._minute_bookkeeping()
        if self.tick_in_hour >= self.c.ticks_per_hour:
            self.tick_in_hour = 0
            self.hour = (self.hour + 1) % self.c.hours_per_day
            if self.c.log_every_hour: self._log_snapshot()
            if self.hour == 0:
                self._end_of_day()
                self._log_snapshot()

    def _mark_death_m(self, man_id:int):
        self.deaths_today += 1; self.deaths_today_m += 1; self.total_deaths += 1; self.total_deaths_m += 1
        self._log_event("death", "M", man_id)
    def _mark_death_w(self, woman_id:int):
        self.deaths_today += 1; self.deaths_today_w += 1; self.total_deaths += 1; self.total_deaths_w += 1
        self._log_event("death", "W", woman_id)

    def _minute_bookkeeping(self):
        cafe=set(self.world.cafe.cells)
        # lethal minimum for women
        for w in self.women:
            if w.dead: continue
            if w.beauty <= self.c.beauty_bounds[0]:
                w.dead=True; self._log_event("death_beauty_min","W",w.id); self._mark_death_w(w.id)
        for m in self.men:
            if m.dead: continue
            if m.pos in m.home.cells: m.home_minutes_today += 1
            if m.pos in cafe:
                m.cafe_session_min += 1; m.cafe_idle_min += 1; m.minutes_since_any_interaction += 1
            else:
                m.cafe_session_min = m.cafe_idle_min = m.minutes_since_any_interaction = 0
            if not m.isolated and m.confidence <= self.c.isolation_conf_threshold:
                m.isolated = True; m.isolation_start_day = self.day_count; m.target = self.world.random_cell(m.home)
        for w in self.women:
            if w.dead: continue
            if w.pos in w.home.cells: w.home_minutes_today += 1
            if w.pos in cafe:
                w.cafe_session_min += 1; w.cafe_idle_min += 1; w.minutes_since_any_interaction += 1
            else:
                w.cafe_session_min = w.cafe_idle_min = w.minutes_since_any_interaction = 0
            if not w.isolated and w.sociality() <= self.c.isolation_conf_threshold:
                w.isolated = True; w.isolation_start_day = self.day_count; w.target = self.world.random_cell(w.home)

    def _end_of_day(self):
        # age and drift
        for m in self.men:
            if m.dead: continue
            m.age_days += 1
            m.confidence = clamp(m.confidence + self.c.conf_daily_drift, *self.c.conf_bounds)
            m.home_minutes_today = 0
        for w in self.women:
            if w.dead: continue
            w.age_days += 1
            w.home_minutes_today = 0

        # beauty decay for women with no match that day
        matched_ids = {w.id for w in self.women if not w.dead and self.day_count==w.last_match_day}
        for w in self.women:
            if w.dead: continue
            if w.id not in matched_ids:
                w.beauty = clamp(w.beauty*(1.0 - self.c.beauty_decay_no_match_per_day), *self.c.beauty_bounds)
            # lethal check after decay
            if w.beauty <= self.c.beauty_bounds[0]:
                w.dead=True; self._log_event("death_beauty_min","W",w.id); self._mark_death_w(w.id)

        # isolation death
        for m in self.men:
            if m.dead: continue
            if m.isolated and (self.day_count - m.isolation_start_day) >= self.c.isolation_death_days:
                m.dead = True; self._log_event("death_isolation","M",m.id); self._mark_death_m(m.id)
        for w in self.women:
            if w.dead: continue
            if w.isolated and (self.day_count - w.isolation_start_day) >= self.c.isolation_death_days:
                w.dead = True; self._log_event("death_isolation","W",w.id); self._mark_death_w(w.id)

        # natural mortality
        self._apply_natural_mortality()

        # purge (do not recompute deaths, we already tracked them)
        self.men=[m for m in self.men if not m.dead]
        self.women=[w for w in self.women if not w.dead]

        # history
        alive = len(self.men)+len(self.women)
        self.days_hist.append(self.day_count)
        self.hist_pop.append(alive)
        self.hist_births.append(self.births_today)
        self.hist_deaths.append(self.deaths_today)
        self.hist_births_m.append(self.births_today_m)
        self.hist_births_w.append(self.births_today_w)
        self.hist_deaths_m.append(self.deaths_today_m)
        self.hist_deaths_w.append(self.deaths_today_w)
        self.hist_matches.append(self.matches_today)
        self.hist_rejects.append(self.rejects_today)
        for arr in (self.days_hist,self.hist_pop,self.hist_births,self.hist_deaths,
                    self.hist_births_m,self.hist_births_w,self.hist_deaths_m,self.hist_deaths_w,
                    self.hist_matches,self.hist_rejects):
            if len(arr)>60: del arr[0:len(arr)-60]

        # reset counters
        self.day_count += 1
        self.matches_today = self.rejects_today = 0
        self.births_today = self.deaths_today = 0
        self.births_today_m = self.births_today_w = 0
        self.deaths_today_m = self.deaths_today_w = 0
        self._update_top_threshold()

    def _apply_natural_mortality(self):
        for m in self.men:
            if m.dead or m.age_days <= self.c.natural_mortality_age_days: continue
            hazard = self.c.natural_mortality_base + self.c.natural_mortality_age_k*(m.age_days - self.c.natural_mortality_age_days)
            if m.health<0.25 or m.confidence<0.15: hazard += self.c.natural_mortality_low_attr_bonus
            hazard = clamp(hazard,0,1)
            if random.random()<hazard: m.dead=True; self._log_event("death_natural","M",m.id); self._mark_death_m(m.id)
        for w in self.women:
            if w.dead or w.age_days <= self.c.natural_mortality_age_days: continue
            hazard = self.c.natural_mortality_base + self.c.natural_mortality_age_k*(w.age_days - self.c.natural_mortality_age_days)
            if w.beauty<0.20 or w.sociality()<0.20: hazard += self.c.natural_mortality_low_attr_bonus
            hazard = clamp(hazard,0,1)
            if random.random()<hazard: w.dead=True; self._log_event("death_natural","W",w.id); self._mark_death_w(w.id)

    # ---- Schedules ----
    def _in_window(self, now:int, start:int, end:int)->bool:
        if start<=end: return start<=now<end
        return now>=start or now<end
    def _sleep_window(self, now:int)->bool:
        return self._in_window(now, 1*60+30, 8*60+30)
    def _work_window(self, now:int, shift:str)->bool:
        return self._in_window(now, 9*60, 17*60) if shift=="MORNING" else self._in_window(now, 17*60, 1*60)

    def desired_building(self, is_man:bool, shift:str, proxy:float,
                         home_mins:int, cafe_session:int, cafe_idle:int, iso:bool) -> Optional[Building]:
        if iso: return None
        now = self._minutes_now()
        if self._sleep_window(now): return None
        if self._work_window(now, shift): return self.world.work
        if cafe_session >= self.c.cafe_session_cap_min: return None
        if cafe_idle >= self.c.cafe_idle_cap_min: return None
        if home_mins < self.c.min_home_minutes_per_day: return None
        threshold = 0.55 - 0.10*(proxy-0.5)
        return self.world.cafe if proxy >= threshold else None

    def _update_top_threshold(self):
        scores = sorted([m.score() for m in self.men], reverse=True)
        if not scores: self.current_threshold=1.0; return
        p = clamp(self.women_accept_top_percent, 0.01, 1.0)
        k = max(1, int(math.ceil(p*len(scores))))
        self.current_threshold = scores[min(k-1, len(scores)-1)]

    def step_people(self):
        for m in self.men:
            if m.dead: continue
            if m.lock_ticks>0: m.lock_ticks-=1; continue
            dest = self.desired_building(True, m.shift, m.confidence, m.home_minutes_today, m.cafe_session_min, m.cafe_idle_min, m.isolated)
            if dest is None:
                if m.target not in m.home.cells or self.tick_in_hour==0: m.target=self.world.random_cell(m.home)
            else:
                if m.target not in dest.cells or self.tick_in_hour==0: m.target=self.world.random_cell(dest)
            self._step_move(m)

        for w in self.women:
            if w.dead: continue
            if w.lock_ticks>0: w.lock_ticks-=1; continue
            soci = w.sociality()
            dest = self.desired_building(False, w.shift, soci, w.home_minutes_today, w.cafe_session_min, w.cafe_idle_min, w.isolated)
            if dest is None:
                if w.target not in w.home.cells or self.tick_in_hour==0: w.target=self.world.random_cell(w.home)
            else:
                if w.target not in dest.cells or self.tick_in_hour==0: w.target=self.world.random_cell(dest)
            self._step_move(w)

    def _step_move(self, person):
        x,y = person.pos; tx,ty = person.target
        if (x,y)==(tx,ty): return
        nx = x + (1 if tx>x else -1 if tx<x else 0)
        ny = y if nx!=x else y + (1 if ty>y else -1 if ty<y else 0)
        if within(self.c, (nx,ny)): person.pos=(nx,ny)

    def try_interactions(self):
        cafe=set(self.world.cafe.cells)
        free_women=[w for w in self.women if not w.dead and w.lock_ticks==0 and w.pos in cafe and not w.isolated]
        if not free_women: return
        for m in self.men:
            if m.dead or m.isolated or m.lock_ticks>0 or m.pos not in cafe: continue
            target=None; best=999
            for w in free_women:
                d=manhattan(m.pos,w.pos)
                if d<=self.c.interaction_radius and d<best: best=d; target=w
            if not target: continue
            mid=((m.pos[0]+target.pos[0])//2, (m.pos[1]+target.pos[1])//2)
            m.target=mid; target.target=mid
            m.lock_ticks=self.c.interaction_lock_ticks
            target.lock_ticks=self.c.interaction_lock_ticks
            self._resolve_interaction(m, target)

    def _resolve_interaction(self, man: Man, woman: Woman):
        hhmm = self._hhmm()
        man_accepts = woman.beauty >= clamp(self.men_beauty_threshold, 0.0, 1.0)
        woman_accepts = man.score() >= self.current_threshold
        if man_accepts and woman_accepts:
            self.matches_today+=1; self.total_matches+=1
            man.last_match_day=self.day_count; woman.last_match_day=self.day_count
            man.confidence = clamp(man.confidence + self.c.conf_gain_on_accept, *self.c.conf_bounds)
            woman.beauty = clamp(woman.beauty + self.c.beauty_recovery_on_match, *self.c.beauty_bounds)
            man.last_result="accept"; woman.last_result="accept"
            man.accepts_total+=1; woman.accepts_total+=1
            man.history.append((self.day_count, hhmm, "accept"))
            woman.history.append((self.day_count, hhmm, "accept"))
            if len(man.history)>10: man.history=man.history[-10:]
            if len(woman.history)>10: woman.history=woman.history[-10:]
            man.cafe_idle_min=woman.cafe_idle_min=0
            man.minutes_since_any_interaction=woman.minutes_since_any_interaction=0
            self._maybe_birth(woman)
        else:
            self.rejects_today+=1; self.total_rejects+=1
            man.confidence = clamp(man.confidence - self.c.conf_loss_on_reject, *self.c.conf_bounds)
            man.last_result="reject"; woman.last_result="reject"
            man.rejects_total+=1; woman.rejects_total+=1
            man.history.append((self.day_count, hhmm, "reject"))
            woman.history.append((self.day_count, hhmm, "reject"))
            if len(man.history)>10: man.history=man.history[-10:]
            if len(woman.history)>10: woman.history=woman.history[-10:]
            man.cafe_idle_min=woman.cafe_idle_min=0
            man.minutes_since_any_interaction=woman.minutes_since_any_interaction=0
            # reposition man to explore
            far = [c for c in self.world.cafe.cells if manhattan(c, man.pos)>=5]
            man.target = random.choice(far or self.world.cafe.cells)
            # isolation trigger
            if not man.isolated and man.confidence <= self.c.isolation_conf_threshold:
                man.isolated = True
                man.isolation_start_day = self.day_count
                man.target = self.world.random_cell(man.home)

    def _maybe_birth(self, woman: Woman):
        if woman.accepts_total==0 or woman.accepts_total % self.c.accepts_per_child != 0:
            return
        if (self.day_count - getattr(woman, "last_birth_day", -9999)) < self.c.birth_cooldown_days:
            return
        alive_total = len(self.men)+len(self.women)
        if alive_total >= self.c.soft_cap_total: return
        ratio = alive_total / max(1, self.c.target_population_total)
        pressure = max(0.0, ratio - 1.0)
        p = self.c.base_birth_prob / (1.0 + self.c.pressure_k*pressure)
        p = clamp(p, 0.0, 1.0)
        if random.random() < p:
            self._spawn_child()
            woman.last_birth_day = self.day_count
            woman.births += 1

    def _spawn_child(self):
        if random.random()<0.5:
            home=random.choice(self.world.men_homes)
            pos=self.world.random_cell(home)
            shift="MORNING" if random.random()<0.5 else "AFTERNOON"
            new_id = (max([m.id for m in self.men])+1) if self.men else 1
            child=Man(id=new_id,pos=pos,home=home,shift=shift,target=pos,
                      money=random.uniform(0.2,0.6),power=random.uniform(0.2,0.6),
                      health=random.uniform(0.5,0.9),confidence=random.uniform(0.3,0.7))
            self.men.append(child); self._log_event("birth","M",child.id)
            self.births_today += 1; self.total_births += 1
            self.births_today_m += 1; self.total_births_m += 1
        else:
            home=random.choice(self.world.women_homes)
            pos=self.world.random_cell(home)
            shift="MORNING" if random.random()<0.5 else "AFTERNOON"
            new_id = (max([w.id for w in self.women])+1) if self.women else 1
            child=Woman(id=new_id,pos=pos,home=home,shift=shift,target=pos,beauty=random.uniform(0.4,0.8))
            self.women.append(child); self._log_event("birth","W",child.id)
            self.births_today += 1; self.total_births += 1
            self.births_today_w += 1; self.total_births_w += 1
        self._update_top_threshold()

    # ---- Logging ----
    def _log_event(self, etype:str, kind:str, pid:int):
        try:
            with open(self.c.log_events_path,"a",encoding="utf-8") as f:
                ts=time.strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"{ts}\tday={self.day_count}\thour={self.hour}\t{etype}\t{kind}{pid}\n")
        except Exception:
            pass

    def _log_snapshot(self):
        try:
            rotate_if_big(self.c.log_csv_path, self.c.log_rotate_max_kb)
            ts=time.strftime("%Y-%m-%d %H:%M:%S")
            minute=int(self.tick_in_hour*60/self.c.ticks_per_hour)
            alive_m=len(self.men); alive_w=len(self.women); alive=alive_m+alive_w
            cafe=set(self.world.cafe.cells); work=set(self.world.work.cells)
            at_cafe=sum(1 for a in [*self.men,*self.women] if a.pos in cafe)
            at_work=sum(1 for a in [*self.men,*self.women] if a.pos in work)
            at_home=alive-at_cafe-at_work
            avg_score=sum(m.score() for m in self.men)/max(1,alive_m)
            avg_conf=sum(m.confidence for m in self.men)/max(1,alive_m)
            avg_beauty=sum(w.beauty for w in self.women)/max(1,alive_w)
            with open(self.c.log_csv_path,"a",newline="",encoding="utf-8") as f:
                csv.writer(f).writerow([
                    ts,self.day_count,self.hour,minute,
                    alive_m,alive_w,alive,
                    round(self.women_accept_top_percent,4), round(self.men_beauty_threshold,4), round(self.current_threshold,4),
                    self.matches_today,self.rejects_today,
                    self.births_today,self.births_today_m,self.births_today_w,
                    self.deaths_today,self.deaths_today_m,self.deaths_today_w,
                    self.total_matches,self.total_rejects,
                    self.total_births,self.total_births_m,self.total_births_w,
                    self.total_deaths,self.total_deaths_m,self.total_deaths_w,
                    round(avg_score,4),round(avg_conf,4),round(avg_beauty,4),
                    at_home,at_work,at_cafe
                ])
        except Exception:
            pass

# =====================================================================================
# Sprites (idle vs interaction icons)
# =====================================================================================

class SpriteFactory:
    def __init__(self, cfg: Config):
        self.c=cfg; self.cache: Dict[str, pygame.Surface]={}

    def tile(self, color)->pygame.Surface:
        key=f"tile_{color}"
        if key in self.cache: return self.cache[key]
        s=self.c.cell; surf=pygame.Surface((s,s), pygame.SRCALPHA)
        pygame.draw.rect(surf, color, (1,1,s-2,s-2), border_radius=5)
        pygame.draw.rect(surf, (0,0,0,90), (1,1,s-2,s-2), 1, border_radius=5)
        self.cache[key]=surf; return surf

    # --- ICONS ---
    def man_idle(self)->pygame.Surface:
        key="man_idle"
        if key in self.cache: return self.cache[key]
        s=self.c.cell; r=s//4
        surf=pygame.Surface((s,s), pygame.SRCALPHA)
        pygame.draw.circle(surf, Colors.ACC1, (s//2, s//2), r)
        pygame.draw.circle(surf, (0,0,0,120), (s//2, s//2), r, 1)
        self.cache[key]=surf; return surf

    def woman_idle(self)->pygame.Surface:
        key="woman_idle"
        if key in self.cache: return self.cache[key]
        s=self.c.cell; r=s//4
        surf=pygame.Surface((s,s), pygame.SRCALPHA)
        pygame.draw.circle(surf, Colors.ACC2, (s//2, s//2), r)
        pygame.draw.circle(surf, (0,0,0,120), (s//2, s//2), r, 1)
        self.cache[key]=surf; return surf

    def man_interact(self)->pygame.Surface:
        key="man_interact"
        if key in self.cache: return self.cache[key]
        s=self.c.cell; r=s//3
        surf=pygame.Surface((s,s), pygame.SRCALPHA)
        pygame.draw.circle(surf, Colors.ACC1, (s//2, s//2), r, 3)        # ring
        pygame.draw.circle(surf, Colors.FG0, (s//2, s//2), 2)             # center dot
        self.cache[key]=surf; return surf

    def woman_interact(self)->pygame.Surface:
        key="woman_interact"
        if key in self.cache: return self.cache[key]
        s=self.c.cell; r=s//3
        surf=pygame.Surface((s,s), pygame.SRCALPHA)
        pygame.draw.circle(surf, Colors.ACC2, (s//2, s//2), r, 3)         # ring (orange)
        pygame.draw.circle(surf, Colors.FG0, (s//2, s//2), 2)
        self.cache[key]=surf; return surf

    def bang(self)->pygame.Surface:
        key="bang"
        if key in self.cache: return self.cache[key]
        s=self.c.cell; surf=pygame.Surface((s,s), pygame.SRCALPHA)
        font=pygame.font.SysFont(None, int(self.c.cell*0.9), bold=True)
        txt=font.render("!", True, Colors.FG0)
        surf.blit(txt, (s//2-txt.get_width()//2, s//2-txt.get_height()//2))
        self.cache[key]=surf; return surf

# =====================================================================================
# UI
# =====================================================================================

class UI:
    def __init__(self, screen: pygame.Surface, fonts: Dict[str, pygame.font.Font]):
        self.screen=screen
        self.fxs=fonts["xs"]; self.fs=fonts["s"]; self.fm=fonts["m"]; self.fl=fonts["l"]

    def card(self, x:int, y:int, w:int, h:int, title:str)->pygame.Rect:
        rect=pygame.Rect(x,y,w,h)
        shadow=pygame.Surface((w,h), pygame.SRCALPHA); shadow.fill((0,0,0,110))
        self.screen.blit(shadow, (x+2,y+2))
        pygame.draw.rect(self.screen, Colors.BG3, rect, border_radius=10)
        header=pygame.Rect(x,y,w,34)
        pygame.draw.rect(self.screen, Colors.BG4, header, border_radius=10)
        pygame.draw.rect(self.screen, Colors.BG4, (x, y+24, w, 10))
        pygame.draw.rect(self.screen, (0,0,0,90), rect, 1, border_radius=10)
        title_surf=self.fm.render(title, True, Colors.ACC2)  # ORANGE HEADERS
        self.screen.blit(title_surf, (x+12, y+8))
        return rect

    def kv(self, x:int, y:int, k:str, v:str, col=Colors.FG1, kpad=210):
        self.screen.blit(self.fs.render(k, True, Colors.FG2), (x,y))
        self.screen.blit(self.fs.render(v, True, col), (x+kpad, y))

    def ratio(self, x:int, y:int, a:int, b:int, label:str, kpad=150):
        total=a+b; r=a/(total+1e-9) if total>0 else 0.0
        barw=220; barh=12
        self.screen.blit(self.fs.render(label, True, Colors.FG2), (x,y-2))
        pygame.draw.rect(self.screen, Colors.BG2, (x+kpad,y,barw,barh), border_radius=6)
        pygame.draw.rect(self.screen, Colors.ACC3, (x+kpad,y,int(barw*r),barh), border_radius=6)
        self.screen.blit(self.fxs.render(f"{a}:{b}", True, Colors.FG2),(x+kpad+barw+8,y-2))

    def _y_ticks(self, x:int, y:int, w:int, h:int, ymin:float, ymax:float, n:int=5):
        for i in range(n+1):
            frac=i/n
            gy=y+24+int((h-38)*frac)
            pygame.draw.line(self.screen, Colors.BG4, (x+10,gy),(x+w-10,gy),1)
            val=ymin + (ymax-ymin)*frac
            lbl=self.fxs.render(f"{val:.0f}", True, Colors.FG2)
            self.screen.blit(lbl, (x+10-lbl.get_width()-6, gy-7))

    def line_chart(self, x:int, y:int, w:int, h:int, data:List[float], color, title:str, xlabels:Optional[List[int]]=None):
        rect=pygame.Rect(x,y,w,h)
        pygame.draw.rect(self.screen, Colors.BG3, rect, border_radius=10)
        pygame.draw.rect(self.screen, (0,0,0,90), rect, 1, border_radius=10)
        self.screen.blit(self.fs.render(title, True, Colors.FG1),(x+12,y+10))
        if not data: return
        vals=list(data)[-min(120,len(data)):]
        n=len(vals); ymin=min(vals); ymax=max(vals)
        if ymax<=ymin: ymax=ymin+1
        self._y_ticks(x,y,w,h,ymin,ymax,5)
        # x-axis labels (every ~6th)
        if xlabels:
            xs=list(xlabels)[-n:]
            step=max(1,n//6)
            basey=y+h-12
            for i in range(0,n,step):
                px=x+12+int((w-24)*i/max(1,n-1))
                lbl=self.fxs.render(str(xs[i]), True, Colors.FG2)
                self.screen.blit(lbl,(px-lbl.get_width()//2, basey-2))
        def N(v): return 0 if ymax==ymin else (v-ymin)/(ymax-ymin)
        pts=[]
        for i,v in enumerate(vals):
            px=x+12+int((w-24)*i/max(1,n-1))
            py=y+h-14 - int((h-38)*N(v))
            pts.append((px,py))
        if len(pts)>=2:
            pygame.draw.aalines(self.screen, color, False, pts)

    def line_chart_multi(self, x:int, y:int, w:int, h:int,
                         series:List[Tuple[List[float], Tuple[int,int,int], str]],
                         title:str, xlabels:Optional[List[int]]=None):
        rect=pygame.Rect(x,y,w,h)
        pygame.draw.rect(self.screen, Colors.BG3, rect, border_radius=10)
        pygame.draw.rect(self.screen, (0,0,0,90), rect, 1, border_radius=10)
        self.screen.blit(self.fs.render(title, True, Colors.FG1),(x+12,y+10))
        if not series: return
        # align lengths; compute min/max
        trimmed=[]
        for data, col, _ in series:
            vals=list(data)[-min(120,len(data)):]
            trimmed.append((vals,col))
        all_vals=[v for vals,_ in trimmed for v in vals] or [0,1]
        ymin=min(all_vals); ymax=max(all_vals)
        if ymax<=ymin: ymax=ymin+1
        self._y_ticks(x,y,w,h,ymin,ymax,5)
        # x labels from the longest series
        n=max(len(vals) for vals,_ in trimmed)
        if xlabels:
            xs=list(xlabels)[-n:]
            step=max(1,n//6)
            basey=y+h-12
            for i in range(0,n,step):
                px=x+12+int((w-24)*i/max(1,n-1))
                lbl=self.fxs.render(str(xs[i]), True, Colors.FG2)
                self.screen.blit(lbl,(px-lbl.get_width()//2, basey-2))
        def N(v): return 0 if ymax==ymin else (v-ymin)/(ymax-ymin)
        for vals,col in trimmed:
            n=len(vals);
            pts=[]
            for i,v in enumerate(vals):
                px=x+12+int((w-24)*i/max(1,n-1))
                py=y+h-14 - int((h-38)*N(v))
                pts.append((px,py))
            if len(pts)>=2:
                pygame.draw.aalines(self.screen, col, False, pts)

# =====================================================================================
# Simulator
# =====================================================================================

class Simulator:
    def __init__(self, cfg: Config):
        self.c=cfg; self.rng=random.Random(cfg.seed)
        pygame.init(); pygame.display.set_caption(cfg.caption)
        self.screen=pygame.display.set_mode((cfg.width, cfg.height))
        self.clock=pygame.time.Clock()
        # +10% fonts
        self.fonts={
            "xs": pygame.font.SysFont(None, 16),
            "s" : pygame.font.SysFont(None, 18),
            "m" : pygame.font.SysFont(None, 20, bold=True),
            "l" : pygame.font.SysFont(None, 24, bold=True),
        }
        self.ui=UI(self.screen, self.fonts)
        self.world=World(cfg, self.rng)
        self.pop=Population(cfg, self.world, self.rng)
        self.sprites=SpriteFactory(cfg)

        self.running=True; self.paused=False; self.fast_skip=0
        self.selected: Optional[Tuple[str,int]]=None
        self.panel_page = self.c.panel_page
        self.panel_scroll=0; self.panel_content_h=0
        self.toast=""; self.toast_ttl=0

        self.tiles={
            "men": self.sprites.tile(Colors.HOME_M),
            "women": self.sprites.tile(Colors.HOME_W),
            "work": self.sprites.tile(Colors.WORK),
            "cafe": self.sprites.tile(Colors.CAFE),
        }

    def run(self):
        while self.running:
            self.handle_events()
            for _ in range(self.c.tpf if not self.paused else 0):
                self.tick()
            if not self.c.fast_forward or self.fast_skip%3==0: self.render()
            self.fast_skip+=1; self.clock.tick(self.c.fps_limit)
        pygame.quit()

    def tick(self):
        self.pop.step_people()
        self.pop.try_interactions()
        self.pop.advance_tick()
        if self.toast_ttl>0: self.toast_ttl-=1

    # ---- Events ----
    def handle_events(self):
        for e in pygame.event.get():
            if e.type==pygame.QUIT:
                self.running=False
            elif e.type==pygame.KEYDOWN:
                k=e.key
                if k==pygame.K_q: self.running=False
                elif k==pygame.K_SPACE: self.paused = not self.paused
                elif k==pygame.K_r: self._reset()
                elif k in (pygame.K_PLUS, pygame.K_EQUALS): self.c.tpf = min(30, self.c.tpf + 1)
                elif k==pygame.K_MINUS: self.c.tpf = max(1, self.c.tpf - 1)
                elif k==pygame.K_g: self.c.draw_grid = not self.c.draw_grid
                elif k==pygame.K_h: self.c.show_help = not self.c.show_help
                elif k==pygame.K_f: self.c.fast_forward = not self.c.fast_forward
                elif k==pygame.K_o: self.c.show_leaderboard = not self.c.show_leaderboard
                elif k==pygame.K_s: self._save()
                elif k==pygame.K_l: self._load()
                elif k==pygame.K_n: self._select_next(+1)
                elif k==pygame.K_b: self._select_next(-1)
                elif k==pygame.K_TAB: self.panel_page = 1 + (self.panel_page % 3); self.panel_scroll = 0
                elif k==pygame.K_1: self.panel_page = 1; self.panel_scroll = 0
                elif k==pygame.K_2: self.panel_page = 2; self.panel_scroll = 0
                elif k==pygame.K_3: self.panel_page = 3; self.panel_scroll = 0
                elif k==pygame.K_LEFTBRACKET:  self._adj_women(-0.01)   # [
                elif k==pygame.K_RIGHTBRACKET: self._adj_women(+0.01)   # ]
                elif k==pygame.K_SEMICOLON:    self._adj_men(-0.05)     # ;
                elif k==pygame.K_QUOTE:        self._adj_men(+0.05)     # '
                elif k==pygame.K_j: self.pop._log_snapshot(); self._toast("Snapshot logged")
                elif k==pygame.K_k: self.c.log_every_hour = not self.c.log_every_hour; self._toast(f"Hourly log {'ON' if self.c.log_every_hour else 'OFF'}")
                elif k==pygame.K_UP:
                    self.panel_scroll = max(0, self.panel_scroll - self.c.panel_scroll_speed)
                elif k==pygame.K_DOWN:
                    lim = max(0, self.panel_content_h - self.c.height + 6)
                    self.panel_scroll = min(lim, self.panel_scroll + self.c.panel_scroll_speed)
            elif e.type == pygame.MOUSEBUTTONDOWN:
                if e.button == 1:
                    self._click(e.pos)
                elif e.button == 4:
                    self.panel_scroll = max(0, self.panel_scroll - self.c.panel_scroll_speed)
                elif e.button == 5:
                    lim = max(0, self.panel_content_h - self.c.height + 6)
                    self.panel_scroll = min(lim, self.panel_scroll + self.c.panel_scroll_speed)

    def _reset(self):
        self.world = World(self.c, self.rng)
        self.pop = Population(self.c, self.world, self.rng)
        self.selected = None
        self.panel_scroll = 0
        self._toast("Reset")

    def _adj_women(self, d: float):
        self.pop.women_accept_top_percent = clamp(self.pop.women_accept_top_percent + d, 0.01, 1.0)
        self.pop._update_top_threshold()
        self._toast(f"Women accept top {int(self.pop.women_accept_top_percent*100)}%")

    def _adj_men(self, d: float):
        self.pop.men_beauty_threshold = clamp(self.pop.men_beauty_threshold + d, 0.0, 1.0)
        self._toast(f"Men beauty threshold {self.pop.men_beauty_threshold:.2f}")

    def _save(self):
        try:
            data = {
                "day": self.pop.day_count, "hour": self.pop.hour, "tick": self.pop.tick_in_hour,
                "women_accept_top_percent": self.pop.women_accept_top_percent,
                "men_beauty_threshold": self.pop.men_beauty_threshold,
                "men": [vars_m(m) for m in self.pop.men],
                "women": [vars_w(w) for w in self.pop.women],
                "hist": {
                    "days": self.pop.days_hist, "pop": self.pop.hist_pop,
                    "births": self.pop.hist_births, "deaths": self.pop.hist_deaths,
                    "births_m": self.pop.hist_births_m, "births_w": self.pop.hist_births_w,
                    "deaths_m": self.pop.hist_deaths_m, "deaths_w": self.pop.hist_deaths_w,
                    "matches": self.pop.hist_matches, "rejects": self.pop.hist_rejects
                }
            }
            with open(self.c.save_path, "w", encoding="utf-8") as f:
                json.dump(data, f)
            self._toast("Saved")
        except Exception as ex:
            print("Save error:", ex); self._toast("Save failed")

    def _load(self):
        try:
            if not os.path.exists(self.c.save_path):
                self._toast("No save"); return
            with open(self.c.save_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            men_home = {b.name: b for b in self.world.men_homes}
            women_home = {b.name: b for b in self.world.women_homes}
            self.pop.men = []; self.pop.women = []
            for d in data.get("men", []):
                home = men_home.get(d["home"], self.world.men_homes[(d["id"]-1) % len(self.world.men_homes)])
                self.pop.men.append(Man(**rehydrate_m(d, home)))
            for d in data.get("women", []):
                home = women_home.get(d["home"], self.world.women_homes[(d["id"]-1) % len(self.world.women_homes)])
                self.pop.women.append(Woman(**rehydrate_w(d, home)))
            self.pop.day_count = int(data.get("day", 0))
            self.pop.hour = int(data.get("hour", self.c.start_hour))
            self.pop.tick_in_hour = int(data.get("tick", 0))
            self.pop.women_accept_top_percent = float(data.get("women_accept_top_percent", self.c.top_percent_default))
            self.pop.men_beauty_threshold = float(data.get("men_beauty_threshold", self.c.men_beauty_threshold_default))
            h = data.get("hist", {})
            self.pop.days_hist = list(h.get("days", []))
            self.pop.hist_pop = list(h.get("pop", []))
            self.pop.hist_births = list(h.get("births", []))
            self.pop.hist_deaths = list(h.get("deaths", []))
            self.pop.hist_births_m = list(h.get("births_m", []))
            self.pop.hist_births_w = list(h.get("births_w", []))
            self.pop.hist_deaths_m = list(h.get("deaths_m", []))
            self.pop.hist_deaths_w = list(h.get("deaths_w", []))
            self.pop.hist_matches = list(h.get("matches", []))
            self.pop.hist_rejects = list(h.get("rejects", []))
            self.pop._update_top_threshold()
            self.panel_scroll = 0
            self._toast("Loaded")
        except Exception as ex:
            print("Load error:", ex); self._toast("Load failed")

    def _click(self, pos: Tuple[int, int]):
        x, y = pos
        if x >= self.c.cols * self.c.cell:
            self.selected = None; return
        best = None; bestd2 = 1e12
        r2 = (self.c.cell*0.8)**2
        for m in self.pop.men:
            px, py = grid_to_px(self.c, m.pos); d2 = (px-x)**2 + (py-y)**2
            if d2 < bestd2 and d2 <= r2:
                best = ("M", m.id); bestd2 = d2
        for w in self.pop.women:
            px, py = grid_to_px(self.c, w.pos); d2 = (px-x)**2 + (py-y)**2
            if d2 < bestd2 and d2 <= r2:
                best = ("W", w.id); bestd2 = d2
        self.selected = best
        if best: self.panel_page = 2; self.panel_scroll = 0

    def _select_next(self, step: int):
        if not self.selected:
            if self.pop.men: self.selected = ("M", self.pop.men[0].id)
            elif self.pop.women: self.selected = ("W", self.pop.women[0].id)
            self.panel_page = 2; return
        kind, idx = self.selected
        ids = [a.id for a in (self.pop.men if kind=="M" else self.pop.women)]
        if not ids: self.selected = None; return
        ids.sort(); i = ids.index(idx) if idx in ids else 0; i = (i + step) % len(ids)
        self.selected = (kind, ids[i]); self.panel_page = 2; self.panel_scroll = 0

    def _toast(self, t: str, ttl: int = 180):
        self.toast = t; self.toast_ttl = ttl

    # ---- Rendering ----
    def render(self):
        self.screen.fill(Colors.BG0)
        self._draw_grid_bg()
        self._draw_buildings()
        self._draw_agents()
        self._draw_right_panel()
        if self.c.show_leaderboard: self._draw_leaderboard()
        if self.toast_ttl > 0: self._draw_toast()
        pygame.display.flip()

    def _draw_grid_bg(self):
        for y in range(self.c.rows):
            for x in range(self.c.cols):
                rect = pygame.Rect(x*self.c.cell, y*self.c.cell, self.c.cell, self.c.cell)
                base = Colors.BG1 if (x+y) % 2 == 0 else Colors.BG2
                pygame.draw.rect(self.screen, base, rect)
        if self.c.draw_grid:
            for x in range(self.c.cols+1):
                X = x*self.c.cell; pygame.draw.line(self.screen, Colors.BG4, (X,0), (X,self.c.height))
            for y in range(self.c.rows+1):
                Y = y*self.c.cell; pygame.draw.line(self.screen, Colors.BG4, (0,Y), (self.c.cols*self.c.cell, Y))

    def _draw_buildings(self):
        t_m = self.tiles["men"]; t_w = self.tiles["women"]; t_wk = self.tiles["work"]; t_cf = self.tiles["cafe"]
        for b in self.world.men_homes:
            for c in b.cells: self.screen.blit(t_m, rect_for_cell(self.c, c).topleft)
            self._label_block(b, "Men Homes")
        for b in self.world.women_homes:
            for c in b.cells: self.screen.blit(t_w, rect_for_cell(self.c, c).topleft)
            self._label_block(b, "Women Homes")
        for c in self.world.work.cells: self.screen.blit(t_wk, rect_for_cell(self.c, c).topleft)
        for c in self.world.cafe.cells: self.screen.blit(t_cf, rect_for_cell(self.c, c).topleft)
        self._label_block(self.world.work, "Work 09–17 / 17–01", solid=True)
        self._label_block(self.world.cafe, "Cafe", solid=True)

    def _label_block(self, b: Building, text: str, solid: bool = False):
        minx = min(x for x,_ in b.cells); maxx = max(x for x,_ in b.cells)
        miny = min(y for _,y in b.cells); maxy = max(y for _,y in b.cells)
        cx = ((minx+maxx)//2)*self.c.cell + self.c.cell//2
        cy = ((miny+maxy)//2)*self.c.cell + self.c.cell//2
        s = self.fonts["s"].render(text, True, Colors.FG0)
        pad = 8; w = s.get_width()+pad*2; h = s.get_height()+pad
        rect = pygame.Rect(cx-w//2, cy-h//2, w, h)
        if solid:
            pygame.draw.rect(self.screen, Colors.BG3, rect, border_radius=8)
        else:
            pygame.draw.rect(self.screen, (0,0,0,100), rect, border_radius=8)
            pygame.draw.rect(self.screen, Colors.BG3, rect, 1, border_radius=8)
        self.screen.blit(s, (rect.x+pad, rect.y+pad//2))

    def _draw_agents(self):
        selected_pos = None
        for m in self.pop.men:
            x,y = grid_to_px(self.c, m.pos)
            icon = self.sprites.man_interact() if m.lock_ticks>0 else self.sprites.man_idle()
            self.screen.blit(icon, icon.get_rect(center=(x,y)))
            if m.lock_ticks>0 and m.last_result!="": self._draw_bang(m.pos, m.last_result=="accept")
            if m.isolated: pygame.draw.circle(self.screen, Colors.FG2, (x,y), 2)
            if self.selected == ("M", m.id): selected_pos=m.pos
        for w in self.pop.women:
            x,y = grid_to_px(self.c, w.pos)
            icon = self.sprites.woman_interact() if w.lock_ticks>0 else self.sprites.woman_idle()
            self.screen.blit(icon, icon.get_rect(center=(x,y)))
            if w.lock_ticks>0 and w.last_result!="": self._draw_bang(w.pos, w.last_result=="accept")
            if w.isolated: pygame.draw.circle(self.screen, Colors.FG2, (x,y), 2)
            if self.selected == ("W", w.id): selected_pos=w.pos
        if selected_pos:
            pygame.draw.rect(self.screen, Colors.FG0, rect_for_cell(self.c, selected_pos), 2, border_radius=6)

    def _draw_bang(self, pos: Tuple[int,int], ok: bool):
        rect = rect_for_cell(self.c, pos); self.screen.blit(self.sprites.bang(), rect.topleft)
        pygame.draw.rect(self.screen, Colors.ACC3 if ok else Colors.ACC4, rect, 2, border_radius=4)

    def _draw_right_panel(self):
        x0 = self.c.cols*self.c.cell
        pygame.draw.rect(self.screen, Colors.BG3, (x0,0,self.c.ui_panel_w,self.c.height))
        content_x = x0 + 14; content_w = self.c.ui_panel_w - 28
        top = 42; y = top - self.panel_scroll

        # Tabs
        for i,name in enumerate(("Summary","Selected","Charts"), start=1):
            rx = x0+14+(i-1)*160; r = pygame.Rect(rx,8,150,28)
            pygame.draw.rect(self.screen, Colors.BG4 if self.panel_page==i else Colors.BG2, r, border_radius=8)
            pygame.draw.rect(self.screen, (0,0,0,100), r, 1, border_radius=8)
            label = self.fonts["s"].render(f"{i}. {name}", True, Colors.FG0 if self.panel_page==i else Colors.FG1)
            self.screen.blit(label, (rx+12, 12))

        if self.panel_page == 1: y = self._page_summary(content_x, y, content_w)
        elif self.panel_page == 2: y = self._page_selected(content_x, y, content_w)
        else: y = self._page_charts(content_x, y, content_w)

        if self.c.show_help:
            y += 10; y = self._page_help(content_x, y, content_w)

        self.panel_content_h = (y - (top - self.panel_scroll)) + self.c.panel_scrollpad
        self.panel_content_h = max(self.panel_content_h, self.c.height)

        if self.panel_content_h > self.c.height:
            frac = self.c.height / self.panel_content_h
            sbh = max(60, int(self.c.height * frac))
            lim = self.panel_content_h - self.c.height
            pos_frac = self.panel_scroll / max(1, lim)
            sby = int((self.c.height - sbh) * pos_frac)
            pygame.draw.rect(self.screen, Colors.FG2, (x0+self.c.ui_panel_w-6, sby, 4, sbh), border_radius=3)

        logline = f"Logging: {'hourly ON' if self.c.log_every_hour else 'hourly OFF'}  [J snapshot, K toggle] -> {os.path.basename(self.c.log_csv_path)}"
        s = self.fonts["xs"].render(logline, True, Colors.FG2)
        self.screen.blit(s, (x0+12, self.c.height - s.get_height() - 6))

    def _page_summary(self, x: int, y: int, w: int) -> int:
        # Simulation
        card = self.ui.card(x,y,w,160,"Simulation")
        minute=int(self.pop.tick_in_hour*60/self.c.ticks_per_hour)
        alive_m=len(self.pop.men); alive_w=len(self.pop.women)
        self.ui.kv(x+12,y+46,"Day / Time",f"{self.pop.day_count}    {self.pop.hour:02d}:{minute:02d}")
        self.ui.kv(x+12,y+72,"Alive",f"Men {alive_m}    Women {alive_w}")
        self.ui.kv(x+12,y+98,"Women accept top %",f"{int(self.pop.women_accept_top_percent*100)}%  ([ / ])")
        self.ui.kv(x+12,y+124,"Men beauty threshold",f"{self.pop.men_beauty_threshold:.2f}  (; / ')")
        y += card.height + 10

        # Now
        card = self.ui.card(x,y,w,160,"Now")
        cafe=set(self.world.cafe.cells); work=set(self.world.work.cells)
        agents=[*self.pop.men,*self.pop.women]
        at_cafe=sum(1 for a in agents if a.pos in cafe)
        at_work=sum(1 for a in agents if a.pos in work)
        at_home=alive_m+alive_w-at_cafe-at_work
        avg_score=sum(m.score() for m in self.pop.men)/max(1,alive_m)
        avg_conf=sum(m.confidence for m in self.pop.men)/max(1,alive_m)
        avg_beauty=sum(wm.beauty for wm in self.pop.women)/max(1,alive_w)
        self.ui.kv(x+12,y+46,"At Home / Work / Cafe",f"{at_home} / {at_work} / {at_cafe}")
        self.ui.kv(x+12,y+72,"Avg men score / conf",f"{avg_score:.2f} / {avg_conf:.2f}")
        self.ui.kv(x+12,y+98,"Avg women beauty",f"{avg_beauty:.2f}")
        self.ui.kv(x+12,y+124,"Top-X% threshold",f"{self.pop.current_threshold:.2f}")
        y += card.height + 10

        # Pressure
        alive = alive_m+alive_w; ratio=alive/max(1,self.c.target_population_total)
        pressure=max(0.0, ratio-1.0)
        birth_prob=self.c.base_birth_prob/(1.0+self.c.pressure_k*pressure)
        birth_prob = 0.0 if alive>=self.c.soft_cap_total else birth_prob
        card = self.ui.card(x,y,w,160,"Population Pressure")
        self.ui.kv(x+12,y+46,"Target / Soft cap",f"{self.c.target_population_total} / {self.c.soft_cap_total}")
        self.ui.kv(x+12,y+72,"Alive total / ratio",f"{alive} / {ratio:.2f}")
        self.ui.kv(x+12,y+98,"Birth prob factor",f"{birth_prob:.2f}")
        self.ui.kv(x+12,y+124,"Accepts/child & cooldown",f"{self.c.accepts_per_child} & {self.c.birth_cooldown_days}d")
        y += card.height + 10

        # Daily stats
        card = self.ui.card(x,y,w,160,"Daily Stats")
        self.ui.kv(x+12,y+46,"Matches today",f"{self.pop.matches_today} (Σ {self.pop.total_matches})", Colors.ACC3)
        self.ui.kv(x+12,y+72,"Rejects today",f"{self.pop.rejects_today} (Σ {self.pop.total_rejects})", Colors.ACC4)
        self.ui.kv(x+12,y+98,"Births today", f"{self.pop.births_today}  M:{self.pop.births_today_m}  W:{self.pop.births_today_w}  (Σ {self.pop.total_births})", Colors.ACC3)
        self.ui.kv(x+12,y+124,"Deaths today", f"{self.pop.deaths_today}  M:{self.pop.deaths_today_m}  W:{self.pop.deaths_today_w}  (Σ {self.pop.total_deaths})")
        y += card.height + 10
        return y

    def _history_grid(self, items: List[Tuple[int,int,str]], x:int, y:int):
        # render last up to 10 entries in 2 columns × 5 rows
        latest = items[-10:][::-1]
        col_w = 200; rows = 5
        for idx, rec in enumerate(latest):
            col = idx // rows   # 0 or 1
            row = idx % rows
            dx = x + 12 + col*col_w
            dy = y + row*20
            d, hm, res = rec
            colr = Colors.ACC3 if res=="accept" else Colors.ACC4
            txt = self.fonts["xs"].render(f"D{d} {hm:04d} {res}", True, colr)
            self.screen.blit(txt, (dx, dy))

    def _page_selected(self, x: int, y: int, w: int) -> int:
        if not self.selected:
            card = self.ui.card(x, y, w, 88, "Selected")
            self.screen.blit(self.fonts["s"].render("Click a person. N/B to cycle.", True, Colors.FG2),
                             (x + 12, y + 48))
            return y + card.height + 10

        kind, idx = self.selected
        title = "Selected Man" if kind == "M" else "Selected Woman"
        # Increase height a bit to avoid overlap
        card = self.ui.card(x, y, w, 360, title)
        yy = y + 44

        if kind == "M":
            m = next((z for z in self.pop.men if z.id == idx), None)
            if m:
                self.ui.kv(x + 12, yy, "ID / Shift", f"{m.id} / {m.shift}"); yy += 24
                self.ui.kv(x + 12, yy, "Age days", f"{m.age_days}"); yy += 24
                self.ui.kv(x + 12, yy, "Score", f"{m.score():.2f}"); yy += 24
                self.ui.kv(x + 12, yy, "Money / Power", f"{m.money:.2f} / {m.power:.2f}"); yy += 24
                self.ui.kv(x + 12, yy, "Health / Conf", f"{m.health:.2f} / {m.confidence:.2f}"); yy += 24
                self.ui.kv(x + 12, yy, "Home mins", f"{m.home_minutes_today}"); yy += 24
                self.ui.ratio(x + 12, yy, m.accepts_total, m.rejects_total, "Accept:Reject"); yy += 32
                self.screen.blit(self.fonts["s"].render("History (last 10):", True, Colors.FG1), (x + 12, yy)); yy += 10
                self._history_grid(m.history, x, yy)
        else:
            wv = next((z for z in self.pop.women if z.id == idx), None)
            if wv:
                self.ui.kv(x + 12, yy, "ID / Shift", f"{wv.id} / {wv.shift}"); yy += 24
                self.ui.kv(x + 12, yy, "Age days", f"{wv.age_days}"); yy += 24
                self.ui.kv(x + 12, yy, "Beauty / Social", f"{wv.beauty:.2f} / {wv.sociality():.2f}"); yy += 24
                self.ui.kv(x + 12, yy, "Births", f"{wv.births}"); yy += 24
                self.ui.kv(x + 12, yy, "Home mins", f"{wv.home_minutes_today}"); yy += 24
                self.ui.ratio(x + 12, yy, wv.accepts_total, wv.rejects_total, "Accept:Reject"); yy += 32
                self.screen.blit(self.fonts["s"].render("History (last 10):", True, Colors.FG1), (x + 12, yy)); yy += 10
                self._history_grid(wv.history, x, yy)

        return y + card.height + 10

    def _page_charts(self, x: int, y: int, w: int) -> int:
        ch = 220
        # population
        self.ui.line_chart(x, y, w, ch, self.pop.hist_pop, Colors.FG0, "Alive population (last 60 days)", self.pop.days_hist); y += ch + 10
        # births (M/W)
        self.ui.line_chart_multi(
            x, y, w, ch,
            [
                (self.pop.hist_births_m, Colors.ACC1, "Male"),
                (self.pop.hist_births_w, Colors.ACC2, "Female"),
            ],
            "Births per day by gender", self.pop.days_hist
        ); y += ch + 10
        # deaths (M/W)
        self.ui.line_chart_multi(
            x, y, w, ch,
            [
                (self.pop.hist_deaths_m, Colors.ACC1, "Male"),
                (self.pop.hist_deaths_w, Colors.ACC2, "Female"),
            ],
            "Deaths per day by gender", self.pop.days_hist
        ); y += ch + 10
        return y

    def _page_help(self, x: int, y: int, w: int) -> int:
        lines = [
            "Space pause/resume, R reset, +/- speed",
            "G grid, H help, F fast forward",
            "S/L save/load, O leaderboard overlay",
            "Click agent; N/B next/prev",
            "[ / ] women accept top %,  ; / ' men beauty threshold",
            "J snapshot, K toggle hourly logging",
            "TAB or 1/2/3 to switch panel page",
            "Scroll panel with wheel or Up/Down",
            "Cafe-only approaches; session/idle caps",
            "Isolation if ≤0.05; death after 2 days",
            "Women die at beauty ≤ 0.05",
            "Births use population pressure and cooldown",
            "All knobs in the Config block at top",
        ]
        line_h = 20
        h = 44 + line_h * len(lines) + 16
        card = self.ui.card(x, y, w, h, "Help")
        yy = y + 44
        for ln in lines:
            self.screen.blit(self.fonts["s"].render(ln, True, Colors.FG1), (x + 12, yy))
            yy += line_h
        return y + card.height + 10

    def _draw_leaderboard(self):
        overlay = pygame.Surface((self.c.width, self.c.height), pygame.SRCALPHA); overlay.fill((0, 0, 0, 160))
        self.screen.blit(overlay, (0, 0))
        W = self.c.ui_panel_w + 280; H = 440
        X = (self.c.width - W) // 2; Y = (self.c.height - H) // 2
        rect = pygame.Rect(X, Y, W, H)
        pygame.draw.rect(self.screen, Colors.BG3, rect, border_radius=12)
        pygame.draw.rect(self.screen, (0, 0, 0, 100), rect, 1, border_radius=12)
        self.screen.blit(self.fonts["l"].render("Top-8 Leaderboards", True, Colors.ACC5), (X + 20, Y + 12))

        men = sorted(self.pop.men, key=lambda m: m.score(), reverse=True)[:8]
        women = sorted(self.pop.women, key=lambda w: w.beauty, reverse=True)[:8]
        col1 = X + 20; col2 = X + W // 2 + 20
        self.screen.blit(self.fonts["m"].render("Men by Score", True, Colors.FG1), (col1, Y + 60))
        self.screen.blit(self.fonts["m"].render("Women by Beauty", True, Colors.FG1), (col2, Y + 60))
        # column headers
        hdr = self.fonts["s"].render("Rank  ID   Score  (M/P/H/C)", True, Colors.FG2); self.screen.blit(hdr, (col1, Y+82))
        hdr2= self.fonts["s"].render("Rank  ID   Beauty  Social  Births", True, Colors.FG2); self.screen.blit(hdr2,(col2, Y+82))
        for i in range(8):
            yy = Y + 104 + i * 34
            pygame.draw.line(self.screen, Colors.BG4, (X+12, yy-6),(X+W-12, yy-6),1)
            if i < len(men):
                m = men[i]
                t = f"{i + 1:>2}.  {m.id:<3}  {m.score():.2f}   ({m.money:.2f}/{m.power:.2f}/{m.health:.2f}/{m.confidence:.2f})"
                self.screen.blit(self.fonts["s"].render(t, True, Colors.FG0), (col1, yy))
            if i < len(women):
                wv = women[i]
                t = f"{i + 1:>2}.  {wv.id:<3}  {wv.beauty:.2f}    {wv.sociality():.2f}    {wv.births}"
                self.screen.blit(self.fonts["s"].render(t, True, Colors.FG0), (col2, yy))

    def _draw_toast(self):
        s = self.fonts["m"].render(self.toast, True, Colors.FG0)
        pad = 10; W = s.get_width() + pad * 2; H = s.get_height() + pad
        x = (self.c.cols * self.c.cell - W) // 2; y = 8
        pygame.draw.rect(self.screen, Colors.BG4, (x, y, W, H), border_radius=8)
        pygame.draw.rect(self.screen, (0, 0, 0, 90), (x, y, W, H), 1, border_radius=8)
        self.screen.blit(s, (x + pad, y + pad // 2))

# ---------- Serialization helpers (module-level) ----------
def vars_m(m: Man) -> Dict:
    return {
        "id": m.id, "pos": m.pos, "home": m.home.name, "shift": m.shift,
        "money": m.money, "power": m.power, "health": m.health, "confidence": m.confidence,
        "target": m.target, "lock_ticks": m.lock_ticks, "last_match_day": m.last_match_day,
        "last_result": m.last_result, "home_minutes_today": m.home_minutes_today, "age_days": m.age_days,
        "cafe_session_min": m.cafe_session_min, "cafe_idle_min": m.cafe_idle_min,
        "minutes_since_any_interaction": m.minutes_since_any_interaction,
        "accepts_total": m.accepts_total, "rejects_total": m.rejects_total, "history": m.history,
        "isolated": m.isolated, "isolation_start_day": m.isolation_start_day, "dead": m.dead
    }

def vars_w(w: Woman) -> Dict:
    return {
        "id": w.id, "pos": w.pos, "home": w.home.name, "shift": w.shift,
        "beauty": w.beauty, "target": w.target, "lock_ticks": w.lock_ticks, "last_match_day": w.last_match_day,
        "last_result": w.last_result, "home_minutes_today": w.home_minutes_today, "age_days": w.age_days,
        "cafe_session_min": w.cafe_session_min, "cafe_idle_min": w.cafe_idle_min,
        "minutes_since_any_interaction": w.minutes_since_any_interaction,
        "accepts_total": w.accepts_total, "rejects_total": w.rejects_total, "births": w.births,
        "history": w.history, "isolated": w.isolated, "isolation_start_day": w.isolation_start_day,
        "dead": w.dead
    }

def rehydrate_m(d, home) -> Dict:
    return {
        "id": int(d["id"]), "pos": tuple(d["pos"]), "home": home,
        "shift": str(d.get("shift", "MORNING")), "money": float(d["money"]), "power": float(d["power"]),
        "health": float(d["health"]), "confidence": float(d["confidence"]),
        "target": tuple(d.get("target", d["pos"])), "lock_ticks": int(d.get("lock_ticks", 0)),
        "last_match_day": int(d.get("last_match_day", -9999)), "last_result": str(d.get("last_result", "")),
        "home_minutes_today": int(d.get("home_minutes_today", 0)), "age_days": int(d.get("age_days", 0)),
        "cafe_session_min": int(d.get("cafe_session_min", 0)), "cafe_idle_min": int(d.get("cafe_idle_min", 0)),
        "minutes_since_any_interaction": int(d.get("minutes_since_any_interaction", 0)),
        "accepts_total": int(d.get("accepts_total", 0)), "rejects_total": int(d.get("rejects_total", 0)),
        "history": [tuple(x) for x in d.get("history", [])],
        "isolated": bool(d.get("isolated", False)),
        "isolation_start_day": int(d.get("isolation_start_day", -9999)),
        "dead": bool(d.get("dead", False))
    }

def rehydrate_w(d, home) -> Dict:
    return {
        "id": int(d["id"]), "pos": tuple(d["pos"]), "home": home,
        "shift": str(d.get("shift", "AFTERNOON")), "beauty": float(d["beauty"]),
        "target": tuple(d.get("target", d["pos"])), "lock_ticks": int(d.get("lock_ticks", 0)),
        "last_match_day": int(d.get("last_match_day", -9999)), "last_result": str(d.get("last_result", "")),
        "home_minutes_today": int(d.get("home_minutes_today", 0)), "age_days": int(d.get("age_days", 0)),
        "cafe_session_min": int(d.get("cafe_session_min", 0)), "cafe_idle_min": int(d.get("cafe_idle_min", 0)),
        "minutes_since_any_interaction": int(d.get("minutes_since_any_interaction", 0)),
        "accepts_total": int(d.get("accepts_total", 0)), "rejects_total": int(d.get("rejects_total", 0)),
        "births": int(d.get("births", 0)), "history": [tuple(x) for x in d.get("history", [])],
        "isolated": bool(d.get("isolated", False)),
        "isolation_start_day": int(d.get("isolation_start_day", -9999)),
        "dead": bool(d.get("dead", False))
    }

# =====================================================================================
# Entry
# =====================================================================================

def main():
    cfg = Config()
    random.seed(cfg.seed)
    sim = Simulator(cfg)
    sim.run()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.")
