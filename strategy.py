#!/usr/bin/env python3
"""
Experiment #5771: 6h Donchian(20) breakout + 1d weekly pivot + volume confirmation
HYPOTHESIS: 6h Donchian breakouts aligned with 1d weekly pivot levels (R3/S3 for fade, R4/S4 for breakout) capture institutional order flow. Volume > 1.5x average confirms participation. Designed for 6h timeframe to balance trade frequency (target: 75-150 trades over 4 years) and work in both bull/bear by requiring confluence of structure (pivots), momentum (breakout), and participation (volume). Uses discrete sizing 0.25.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_5771_6h_donchian20_1d_weekly_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute session hours once (open_time is already datetime64[ms])
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # === HTF: 1d data for weekly pivot calculation ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) >= 5:
        # Calculate weekly pivot from prior week (Mon-Sun) using 1d data
        # Need to group by week: find Monday of each week
        # We'll use: for each 1d bar, if it's Sunday, calculate pivot for prior week (Mon-Sun)
        close_1d = df_1d['close'].values
        high_1d = df_1d['high'].values
        low_1d = df_1d['low'].values
        
        # Calculate weekly pivot for each week (using prior completed week)
        # We'll store pivot values aligned to each 1d bar (Sunday close)
        weekly_pivot = np.full(len(df_1d), np.nan)
        weekly_r3 = np.full(len(df_1d), np.nan)
        weekly_s3 = np.full(len(df_1d), np.nan)
        weekly_r4 = np.full(len(df_1d), np.nan)
        weekly_s4 = np.full(len(df_1d), np.nan)
        
        # Group by week: find weeks where we have Mon-Sun data
        # Simpler: for each Sunday, look back 6 days for Mon-Sun
        for i in range(6, len(df_1d)):  # start at index 6 (7th day, assuming index 0 is Monday)
            # Check if we have 7 consecutive days (Mon-Sun)
            # We'll use: if we can look back 6 days, calculate pivot for that week
            week_high = np.max(high_1d[i-6:i+1])  # Mon-Sun high
            week_low = np.min(low_1d[i-6:i+1])    # Mon-Sun low
            week_close = close_1d[i]               # Sunday close
            
            pivot_point = (week_high + week_low + week_close) / 3.0
            weekly_pivot[i] = pivot_point
            
            # Camarilla levels
            range_val = week_high - week_low
            weekly_r3[i] = pivot_point + range_val * 1.1 / 2
            weekly_s3[i] = pivot_point - range_val * 1.1 / 2
            weekly_r4[i] = pivot_point + range_val * 1.1
            weekly_s4[i] = pivot_point - range_val * 1.1
    else:
        weekly_pivot = np.full(len(df_1d), np.nan)
        weekly_r3 = np.full(len(df_1d), np.nan)
        weekly_s3 = np.full(len(df_1d), np.nan)
        weekly_r4 = np.full(len(df_1d), np.nan)
        weekly_s4 = np.full(len(df_1d), np.nan)
    
    # Align weekly pivot levels to 6h timeframe (shifted by 1 for completed 1d bars only)
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    weekly_r3_aligned = align_htf_to_ltf(prices, df_1d, weekly_r3)
    weekly_s3_aligned = align_htf_to_ltf(prices, df_1d, weekly_s3)
    weekly_r4_aligned = align_htf_to_ltf(prices, df_1d, weekly_r4)
    weekly_s4_aligned = align_htf_to_ltf(prices, df_1d, weekly_s4)
    
    # === 6h Indicators: Donchian Channel (20-period) ===
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 6h Indicators: Volume confirmation ===
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / np.where(avg_volume > 0, avg_volume, 1)
    
    # === 6h Indicators: ATR(14) for trailing stop ===
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size (discrete level)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(20, 20, 14, 7)  # Donchian, volume avg, ATR, weekly data
    
    for i in range(warmup, n):
        # --- Session Filter: Avoid low liquidity periods ---
        hour = hours[i]
        if 21 <= hour <= 23:
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_ratio[i]) or np.isnan(atr[i]) or
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(weekly_r3_aligned[i]) or
            np.isnan(weekly_s3_aligned[i]) or np.isnan(weekly_r4_aligned[i]) or
            np.isnan(weekly_s4_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            if position_side > 0:  # Long position
                highest_since_entry = max(highest_since_entry, high[i])
                stop_price = highest_since_entry - 2.5 * atr[i]
                # Exit: stoploss OR price breaks below S3 (fade failed) OR price breaks above R4 (take profit)
                if price <= stop_price or price < weekly_s3_aligned[i] or price >= weekly_r4_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short position
                lowest_since_entry = min(lowest_since_entry, low[i])
                stop_price = lowest_since_entry + 2.5 * atr[i]
                # Exit: stoploss OR price breaks above R3 (fade failed) OR price breaks below S4 (take profit)
                if price >= stop_price or price > weekly_r3_aligned[i] or price <= weekly_s4_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        breakout_up = price > donchian_high[i-1]
        breakout_down = price < donchian_low[i-1]
        volume_confirmed = volume_ratio[i] > 1.5
        
        # Entry conditions:
        # LONG: breakout above R4 with volume (breakout continuation)
        # SHORT: breakout below S4 with volume (breakout continuation)
        # LONG: pullback to S3 with volume (fade at support)
        # SHORT: pullback to R3 with volume (fade at resistance)
        long_breakout = breakout_up and volume_confirmed and price > weekly_r4_aligned[i]
        short_breakout = breakout_down and volume_confirmed and price < weekly_s4_aligned[i]
        long_fade = (price <= weekly_s3_aligned[i] * 1.005) and volume_confirmed and price > weekly_s3_aligned[i] * 0.995  # near S3
        short_fade = (price >= weekly_r3_aligned[i] * 0.995) and volume_confirmed and price <= weekly_r3_aligned[i] * 1.005  # near R3
        
        if long_breakout or long_fade:
            in_position = True
            position_side = 1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = SIZE
        elif short_breakout or short_fade:
            in_position = True
            position_side = -1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals