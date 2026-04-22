#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Load 1d data once
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate weekly Pivot levels using previous week's HLC (no look-ahead)
    # We need to resample to weekly first, but we'll do it manually by taking every 5th day
    # Approximate weekly by taking Friday's data (assuming 5 trading days per week)
    weekly_high = np.full_like(high_1d, np.nan)
    weekly_low = np.full_like(low_1d, np.nan)
    weekly_close = np.full_like(close_1d, np.nan)
    
    # Simple approach: every 5th bar is weekly (approximation)
    for i in range(4, len(high_1d), 5):
        weekly_high[i] = np.max(high_1d[i-4:i+1])
        weekly_low[i] = np.min(low_1d[i-4:i+1])
        weekly_close[i] = close_1d[i]
    
    # Forward fill weekly values
    weekly_high = pd.Series(weekly_high).ffill().bfill().values
    weekly_low = pd.Series(weekly_low).ffill().bfill().values
    weekly_close = pd.Series(weekly_close).ffill().bfill().values
    
    # Calculate weekly Pivot from previous week's data
    prev_weekly_high = np.roll(weekly_high, 1)
    prev_weekly_low = np.roll(weekly_low, 1)
    prev_weekly_close = np.roll(weekly_close, 1)
    prev_weekly_high[0] = np.nan
    prev_weekly_low[0] = np.nan
    prev_weekly_close[0] = np.nan
    
    weekly_pp = (prev_weekly_high + prev_weekly_low + prev_weekly_close) / 3
    weekly_r1 = 2 * weekly_pp - prev_weekly_low
    weekly_s1 = 2 * weekly_pp - prev_weekly_high
    
    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align to 6h timeframe
    weekly_pp_aligned = align_htf_to_ltf(prices, df_1d, weekly_pp)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1d, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1d, weekly_s1)
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume spike filter (20-period average on 6h data)
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):
        # Skip if any data is not ready
        if (np.isnan(weekly_pp_aligned[i]) or 
            np.isnan(weekly_r1_aligned[i]) or 
            np.isnan(weekly_s1_aligned[i]) or 
            np.isnan(ema50_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        pp = weekly_pp_aligned[i]
        r1 = weekly_r1_aligned[i]
        s1 = weekly_s1_aligned[i]
        ema50 = ema50_aligned[i]
        
        if position == 0:
            # Long: price breaks above weekly R1 with volume + above EMA50
            if price > r1 and vol > 2.0 * vol_ma and price > ema50:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly S1 with volume + below EMA50
            elif price < s1 and vol > 2.0 * vol_ma and price < ema50:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: price crosses back through weekly central pivot
            if position == 1 and price < pp:
                signals[i] = 0.0
                position = 0
            elif position == -1 and price > pp:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_WeeklyPivot_R1_S1_Breakout_1dEMA50_Volume_Spike"
timeframe = "6h"
leverage = 1.0