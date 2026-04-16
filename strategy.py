#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d weekly pivot (R1/S1) breakout with volume confirmation and 1w ADX trend filter.
# Long when price > 1d R1 pivot, 6h volume > 1.3x median, and 1w ADX > 25 (trending).
# Short when price < 1d S1 pivot, same volume condition, and 1w ADX > 25.
# Exit when price crosses the 1d daily pivot point.
# Uses discrete position size 0.25. No session filter to capture 6h sessions.
# Target: 80-160 total trades over 4 years (20-40/year). Uses 1d/1w for direction, 6h for timing.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once before loop for pivot points
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === 1d Indicators: Weekly pivot points (using prior week's H/L/C) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate weekly pivot points (using prior week's OHLC)
    # For each 1d bar, use the prior week's (7 days ago) H/L/C
    # We'll use rolling window of 7 to get prior week's values
    prior_week_high = pd.Series(high_1d).rolling(window=7, min_periods=7).max().shift(1).values
    prior_week_low = pd.Series(low_1d).rolling(window=7, min_periods=7).min().shift(1).values
    prior_week_close = pd.Series(close_1d).rolling(window=7, min_periods=7).last().shift(1).values
    
    # Weekly pivot point
    pivot = (prior_week_high + prior_week_low + prior_week_close) / 3.0
    # R1 and S1 levels
    r1 = 2 * pivot - prior_week_low
    s1 = 2 * pivot - prior_week_high
    
    # Get 1w data for ADX trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # === 1w Indicators: ADX(14) for trend strength ===
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = pd.Series(high_1w).shift(1) - pd.Series(low_1w).shift(1)
    tr2 = abs(pd.Series(high_1w) - pd.Series(close_1w).shift(1))
    tr3 = abs(pd.Series(low_1w) - pd.Series(close_1w).shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Movement
    up_move = pd.Series(high_1w) - pd.Series(high_1w).shift(1)
    down_move = pd.Series(low_1w).shift(1) - pd.Series(low_1w)
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed DM
    plus_di = 100 * pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr
    
    # DX and ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Get 6h data for volume confirmation
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 30:
        return np.zeros(n)
    
    # === 6h Indicators: Volume median for spike detection ===
    vol_6h = df_6h['volume'].values
    vol_median_20 = pd.Series(vol_6h).rolling(window=20, min_periods=20).median().values
    
    # Align all indicators to primary timeframe (6h)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    vol_median_aligned = align_htf_to_ltf(prices, df_6h, vol_median_20)
    vol_6h_aligned = align_htf_to_ltf(prices, df_6h, vol_6h)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(7+1, 14+14, 20)  # prior week shift, ADX smoothing, volume median
    
    # Track position state for exits
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(pivot_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(vol_median_aligned[i]) or np.isnan(vol_6h_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values (aligned)
        r1 = r1_aligned[i]
        s1 = s1_aligned[i]
        pivot = pivot_aligned[i]
        adx_val = adx_aligned[i]
        vol_median = vol_median_aligned[i]
        vol_6h = vol_6h_aligned[i]
        
        # Price levels
        price = close[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        if position == 1:  # long position
            # Exit when price crosses below daily pivot (mean reversion to pivot)
            if price < pivot:
                exit_signal = True
        elif position == -1:  # short position
            # Exit when price crosses above daily pivot (mean reversion to pivot)
            if price > pivot:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Volume spike filter: current 6h volume > 1.3x median volume
            volume_spike = vol_6h > (vol_median * 1.3)
            # Trend filter: 1w ADX > 25 indicates trending market
            trending = adx_val > 25
            
            # LONG CONDITIONS
            # Price breaks above R1 pivot AND volume spike AND trending market
            if price > r1 and volume_spike and trending:
                signals[i] = 0.25
                position = 1
            
            # SHORT CONDITIONS
            # Price breaks below S1 pivot AND volume spike AND trending market
            elif price < s1 and volume_spike and trending:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25  # maintain position
    
    return signals

name = "6h_WeeklyPivot_R1S1_Breakout_VolumeSpike1.3x_1wADX25_v1"
timeframe = "6h"
leverage = 1.0