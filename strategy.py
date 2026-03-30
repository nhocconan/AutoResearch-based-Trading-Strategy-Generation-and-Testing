#!/usr/bin/env python3
"""
Experiment #024: 6h Weekly Pivot Breakout + 1w Trend + Donchian Confirmation

HYPOTHESIS: Weekly pivots (from 1w data) create stable institutional reference 
levels. Price breaking a weekly pivot with Donchian(20) confirmation captures 
significant multi-day moves. 1w EMA20 as trend filter ensures we only trade 
with the higher timeframe direction.

WHY 6h: Slower than 4h but faster than 12h. Weekly pivots on 6h = institutional
timeframe. Trade frequency naturally limited by weekly pivot structure.

WHY IT WORKS: Weekly pivots recalculate once per week, not daily. This creates
stable zones. Donchian(20) confirms momentum. 1w trend filter prevents 
counter-trend trades. 6h = ~120 bars/week vs 24 bars on 1d = better resolution.

TARGET: 75-150 total trades over 4 years (19-37/year).
Signal size: 0.30.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_weekly_pivot_donchian_1w_v1"
timeframe = "6h"
leverage = 1.0

def calculate_pivot_levels(high, low, close):
    """Classic pivot point calculation"""
    pivot = (high + low + close) / 3.0
    r1 = 2 * pivot - low
    s1 = 2 * pivot - high
    r2 = pivot + (high - low)
    s2 = pivot - (high - low)
    r3 = high + 2 * (pivot - low)
    s3 = low - 2 * (high - pivot)
    return pivot, r1, s1, r2, s2, r3, s3

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    
    # 1w EMA20 for trend
    ema_1w = pd.Series(df_1w['close'].values).ewm(span=20, min_periods=20, adjust=False).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # 1d EMA50 for shorter trend
    ema_1d = pd.Series(df_1d['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Weekly pivot levels from 1w data
    pivot_1w, r1_1w, s1_1w, r2_1w, s2_1w, r3_1w, s3_1w = calculate_pivot_levels(
        df_1w['high'].values, df_1w['low'].values, df_1w['close'].values
    )
    # Align to 6h
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    
    # === Local 6h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Donchian channel (20 periods = 5 days on 6h)
    donchian_period = 20
    highest_high = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    lowest_low = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    # Volume SMA
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # Signals
    signals = np.zeros(n)
    SIZE = 0.30
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_bar = 0
    
    warmup = 100
    
    for i in range(warmup, n):
        # Skip if ATR not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # Skip if HTF not aligned
        if np.isnan(ema_1w_aligned[i]) or np.isnan(ema_1d_aligned[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === TREND DIRECTION (1w + 1d) ===
        price_above_1w_ema = close[i] > ema_1w_aligned[i]
        price_above_1d_ema = close[i] > ema_1d_aligned[i]
        bull_trend = price_above_1w_ema and price_above_1d_ema
        bear_trend = not price_above_1w_ema and not price_above_1d_ema
        
        # Volume confirmation
        vol_spike = vol_ratio[i] > 1.5
        
        # Weekly pivot levels (use latest available)
        pivot = pivot_aligned[i] if not np.isnan(pivot_aligned[i]) else 0.0
        r1 = r1_aligned[i] if not np.isnan(r1_aligned[i]) else close[i] * 1.01
        s1 = s1_aligned[i] if not np.isnan(s1_aligned[i]) else close[i] * 0.99
        
        # === DONCHIAN BREAKOUT ===
        # Use shift(1) to avoid look-ahead
        donchian_up_broken = close[i] > highest_high[i - 1]
        donchian_down_broken = close[i] < lowest_low[i - 1]
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG ENTRY ===
            # Bull trend + break above weekly R1 or Donchian high + volume
            if bull_trend:
                # Primary: Donchian breakout with volume
                if donchian_up_broken and vol_spike:
                    desired_signal = SIZE
                # Secondary: Price reclaims weekly R1 after pullback with volume
                elif close[i] > r1 and close[i - 1] < r1 and vol_spike:
                    desired_signal = SIZE
            
            # === SHORT ENTRY ===
            # Bear trend + break below weekly S1 or Donchian low + volume
            if bear_trend:
                # Primary: Donchian breakdown with volume
                if donchian_down_broken and vol_spike:
                    desired_signal = -SIZE
                # Secondary: Price breaks below weekly S1 with volume
                elif close[i] < s1 and close[i - 1] > s1 and vol_spike:
                    desired_signal = -SIZE
        
        # === STOPLOSS (3 ATR trailing for 6h) ===
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 3.0 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 3.0 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                desired_signal = 0.0
        
        # === TAKE PROFIT at 2.5R + half position ===
        bars_held = i - entry_bar
        if in_position and bars_held >= 3:
            if position_side > 0:
                profit_2_5r = entry_price + 2.5 * entry_atr
                if high[i] >= profit_2_5r:
                    desired_signal = SIZE * 0.5  # Take half profit
            elif position_side < 0:
                profit_2_5r = entry_price - 2.5 * entry_atr
                if low[i] <= profit_2_5r:
                    desired_signal = -SIZE * 0.5
        
        # === HOLD MINIMUM 3 bars to avoid fee churn ===
        if in_position and bars_held < 3:
            if position_side > 0:
                desired_signal = SIZE
            elif position_side < 0:
                desired_signal = -SIZE
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                # New position or flip
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                entry_bar = i
                if position_side > 0:
                    stop_price = entry_price - 3.0 * entry_atr
                else:
                    stop_price = entry_price + 3.0 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
        
        signals[i] = desired_signal
    
    return signals