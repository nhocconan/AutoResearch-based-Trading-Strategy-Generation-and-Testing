#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Weekly EMA(50) for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Get daily data for weekly pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Calculate weekly pivot points using previous week's OHLC
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w_arr = df_1w['close'].values
    
    # Pivot point = (H + L + C) / 3
    pp_1w = (high_1w + low_1w + close_1w_arr) / 3.0
    # R1 = 2*P - L, S1 = 2*P - H
    r1_1w = 2 * pp_1w - low_1w
    s1_1w = 2 * pp_1w - high_1w
    # R2 = P + (H - L), S2 = P - (H - L)
    r2_1w = pp_1w + (high_1w - low_1w)
    s2_1w = pp_1w - (high_1w - low_1w)
    # R3 = H + 2*(P - L), S3 = L - 2*(H - P)
    r3_1w = high_1w + 2 * (pp_1w - low_1w)
    s3_1w = low_1w - 2 * (high_1w - pp_1w)
    
    # Align weekly pivot levels to 6h
    pp_1w_aligned = align_htf_to_ltf(prices, df_1w, pp_1w)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    r2_1w_aligned = align_htf_to_ltf(prices, df_1w, r2_1w)
    s2_1w_aligned = align_htf_to_ltf(prices, df_1w, s2_1w)
    r3_1w_aligned = align_htf_to_ltf(prices, df_1w, r3_1w)
    s3_1w_aligned = align_htf_to_ltf(prices, df_1w, s3_1w)
    
    # 6h Donchian channels (20-period) for entry timing
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    
    highest_high_6h = pd.Series(high_6h).rolling(window=20, min_periods=20).max().values
    lowest_low_6h = pd.Series(low_6h).rolling(window=20, min_periods=20).min().values
    highest_high_6h_aligned = align_htf_to_ltf(prices, df_6h, highest_high_6h)
    lowest_low_6h_aligned = align_htf_to_ltf(prices, df_6h, lowest_low_6h)
    
    # Volume confirmation: current volume > 1.3x average volume (6h average)
    vol_ma_6h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > vol_ma_6h * 1.3
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20, 20)
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(pp_1w_aligned[i]) or
            np.isnan(r1_1w_aligned[i]) or
            np.isnan(s1_1w_aligned[i]) or
            np.isnan(highest_high_6h_aligned[i]) or
            np.isnan(lowest_low_6h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter from weekly EMA
        uptrend = close[i] > ema_50_1w_aligned[i]
        downtrend = close[i] < ema_50_1w_aligned[i]
        
        # Breakout conditions from 6h Donchian
        breakout_up = close[i] > highest_high_6h_aligned[i]
        breakout_down = close[i] < lowest_low_6h_aligned[i]
        
        # Pivot-based filters: avoid buying too high, selling too low
        # In uptrend: look for pullbacks to S1/S2 for longs
        # In downtrend: look for rallies to R1/R2 for shorts
        pullback_to_support = (close[i] <= s1_1w_aligned[i] * 1.02) or (close[i] <= s2_1w_aligned[i] * 1.02)
        rally_to_resistance = (close[i] >= r1_1w_aligned[i] * 0.98) or (close[i] >= r2_1w_aligned[i] * 0.98)
        
        # Entry conditions: require trend + breakout + volume + pivot alignment
        long_entry = uptrend and breakout_up and volume_confirm[i] and pullback_to_support
        short_entry = downtrend and breakout_down and volume_confirm[i] and rally_to_resistance
        
        # Exit conditions: when trend reverses or opposite breakout
        if position == 1:
            exit_condition = not uptrend or breakout_down
        elif position == -1:
            exit_condition = not downtrend or breakout_up
        else:
            exit_condition = False
        
        # Handle entries and exits
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif exit_condition and position != 0:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_WeeklyEMA50_PivotFilter_Donchian20_Volume"
timeframe = "6h"
leverage = 1.0