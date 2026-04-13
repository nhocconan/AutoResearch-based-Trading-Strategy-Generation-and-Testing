#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Camarilla pivot breakout with volume confirmation and 1d trend filter
    # Long: price breaks above H3 (resistance) AND volume > 1.8x 20-period average AND close > 1d EMA50
    # Short: price breaks below L3 (support) AND volume > 1.8x 20-period average AND close < 1d EMA50
    # Exit: price returns to PIVOT point
    # Using 4h for Camarilla pivots (structure), 1d for trend filter (direction)
    # Higher volume threshold (1.8x) to reduce trades and avoid fee drag
    # Discrete position sizing (0.25) for balance between return and risk
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Get 4h data for Camarilla pivots (call ONCE before loop)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 4h Camarilla pivots (based on previous 4h bar)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # PIVOT = (H + L + C) / 3
    pivot_4h = (high_4h + low_4h + close_4h) / 3
    # RANGE = H - L
    range_4h = high_4h - low_4h
    
    # Camarilla levels:
    # H3 = C + RANGE * 1.1/4
    # L3 = C - RANGE * 1.1/4
    h3_4h = close_4h + range_4h * 1.1 / 4
    l3_4h = close_4h - range_4h * 1.1 / 4
    
    # Align 4h Camarilla levels to 1h (wait for completed 4h bar)
    h3_4h_aligned = align_htf_to_ltf(prices, df_4h, h3_4h)
    l3_4h_aligned = align_htf_to_ltf(prices, df_4h, l3_4h)
    pivot_4h_aligned = align_htf_to_ltf(prices, df_4h, pivot_4h)
    
    # Get 1d data for trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volume confirmation: >1.8x 20-period average (higher threshold to reduce trades)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.8 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(h3_4h_aligned[i]) or np.isnan(l3_4h_aligned[i]) or 
            np.isnan(pivot_4h_aligned[i]) or np.isnan(volume_spike[i]) or
            np.isnan(ema_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume_spike[i]
        
        # Trend filter: only long if close > 1d EMA50, only short if close < 1d EMA50
        long_trend_ok = close[i] > ema_1d_aligned[i]
        short_trend_ok = close[i] < ema_1d_aligned[i]
        
        # Entry logic: Camarilla breakout + volume + trend
        long_entry = (close[i] > h3_4h_aligned[i]) and vol_confirm and long_trend_ok
        short_entry = (close[i] < l3_4h_aligned[i]) and vol_confirm and short_trend_ok
        
        # Exit logic: return to pivot
        long_exit = close[i] < pivot_4h_aligned[i]
        short_exit = close[i] > pivot_4h_aligned[i]
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_1d_camarilla_breakout_volume_trend_v3"
timeframe = "4h"
leverage = 1.0