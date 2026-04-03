#!/usr/bin/env python3
"""
Experiment #1967: 6h Donchian Breakout + Weekly Pivot Direction + Volume Confirmation
HYPOTHESIS: Weekly pivot levels provide institutional structure, while 6h Donchian breakouts capture momentum.
Strategy uses weekly pivot (R1/S1) as bias filter: only take longs when price > weekly pivot, shorts when price < weekly pivot.
Enter on 6h Donchian(20) breakout with volume confirmation (>1.5x 20-period average volume).
Exit when price touches opposite Donchian band or weekly pivot level is crossed.
Target: 75-150 total trades over 4 years (19-37/year). Works in bull/bear by following weekly structure.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_1967_6h_donchian20_1w_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for weekly aggregation ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # === Calculate Weekly Pivot from daily data ===
    # Group into weeks (5 trading days approximate)
    weeks_high = []
    weeks_low = []
    weeks_close = []
    
    for i in range(0, len(high_1d), 5):
        week_high = np.max(high_1d[i:i+5]) if i+5 <= len(high_1d) else np.max(high_1d[i:])
        week_low = np.min(low_1d[i:i+5]) if i+5 <= len(low_1d) else np.min(low_1d[i:])
        week_close = close_1d[min(i+4, len(close_1d)-1)] if i+4 < len(close_1d) else close_1d[-1]
        weeks_high.append(week_high)
        weeks_low.append(week_low)
        weeks_close.append(week_close)
    
    weeks_high = np.array(weeks_high)
    weeks_low = np.array(weeks_low)
    weeks_close = np.array(weeks_close)
    
    # Weekly pivot = (H + L + C) / 3
    weekly_pivot = (weeks_high + weeks_low + weeks_close) / 3.0
    weekly_range = weeks_high - weeks_low
    # Weekly R1 = Pivot + (R1 - S1) where R1-S1 = Range
    weekly_r1 = weekly_pivot + weekly_range
    weekly_s1 = weekly_pivot - weekly_range
    
    # Align weekly data to daily then to 6h
    # First create daily-aligned weekly arrays
    daily_weekly_pivot = np.repeat(weekly_pivot, 5)[:len(high_1d)]
    daily_weekly_r1 = np.repeat(weekly_r1, 5)[:len(high_1d)]
    daily_weekly_s1 = np.repeat(weekly_s1, 5)[:len(high_1d)]
    
    # Pad to full length
    pad_len = len(high_1d) - len(daily_weekly_pivot)
    if pad_len > 0:
        daily_weekly_pivot = np.pad(daily_weekly_pivot, (0, pad_len), mode='edge')
        daily_weekly_r1 = np.pad(daily_weekly_r1, (0, pad_len), mode='edge')
        daily_weekly_s1 = np.pad(daily_weekly_s1, (0, pad_len), mode='edge')
    
    # Align to 6h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, daily_weekly_pivot)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1d, daily_weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1d, daily_weekly_s1)
    
    # === 6h Indicators: Donchian(20) and Volume MA(20) ===
    donchian_h = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_l = pd.Series(low).rolling(window=20, min_periods=20).min().values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 50  # sufficient for Donchian(20) and volume MA
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donchian_h[i]) or np.isnan(donchian_l[i]) or
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(weekly_r1_aligned[i]) or
            np.isnan(weekly_s1_aligned[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            bars_since_entry += 1
            
            # Exit conditions
            exit_signal = False
            
            if position_side > 0:  # Long position
                # Exit if price touches weekly pivot (mean reversion to weekly structure)
                if price <= weekly_pivot_aligned[i]:
                    exit_signal = True
                # Exit if price touches lower Donchian band (breakdown)
                elif price <= donchian_l[i]:
                    exit_signal = True
            else:  # Short position
                # Exit if price touches weekly pivot (mean reversion to weekly structure)
                if price >= weekly_pivot_aligned[i]:
                    exit_signal = True
                # Exit if price touches upper Donchian band (breakout)
                elif price >= donchian_h[i]:
                    exit_signal = True
            
            if exit_signal:
                in_position = False
                position_side = 0
                bars_since_entry = 0
                signals[i] = 0.0
            else:
                signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Volume confirmation: require volume spike (> 1.5x average)
        volume_spike = vol_ratio[i] > 1.5
        
        if volume_spike:
            # Long entry: price breaks above upper Donchian AND price above weekly pivot (bullish bias)
            if price > donchian_h[i] and price > weekly_pivot_aligned[i]:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Short entry: price breaks below lower Donchian AND price below weekly pivot (bearish bias)
            elif price < donchian_l[i] and price < weekly_pivot_aligned[i]:
                in_position = True
                position_side = -1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals