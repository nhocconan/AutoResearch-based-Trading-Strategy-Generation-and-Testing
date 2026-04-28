#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Weekly EMA(50) for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Get daily data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Daily Camarilla pivot levels (H3/L3 for exits, H4/L4 for entries)
    pivot = (high_1d + low_1d + close_1d) / 3
    range_hl = high_1d - low_1d
    H4 = close_1d + (range_hl * 1.1 / 2)
    L4 = close_1d - (range_hl * 1.1 / 2)
    H3 = close_1d + (range_hl * 1.1 / 4)
    L3 = close_1d - (range_hl * 1.1 / 4)
    
    # Align pivot levels to 12h
    H4_aligned = align_htf_to_ltf(prices, df_1d, H4)
    L4_aligned = align_htf_to_ltf(prices, df_1d, L4)
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    
    # Volume confirmation: current volume > 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR-based volatility filter: avoid extreme volatility
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = 0
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_pct = atr / close
    atr_ma = pd.Series(atr_pct).rolling(window=50, min_periods=50).mean().values
    volatility_filter = atr_pct < (atr_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 34, 20, 50)
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(H4_aligned[i]) or 
            np.isnan(L4_aligned[i]) or
            np.isnan(vol_ma_20[i]) or
            np.isnan(H3_aligned[i]) or
            np.isnan(L3_aligned[i]) or
            np.isnan(volatility_filter[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter from weekly EMA(50)
        uptrend = close[i] > ema_50_1w_aligned[i]
        downtrend = close[i] < ema_50_1w_aligned[i]
        
        # Volume filter: current 12h volume above average
        volume_filter = volume[i] > vol_ma_20[i]
        
        # Entry conditions: Camarilla H4/L4 breakout with volume, trend, and volatility filter
        long_breakout = close[i] > H4_aligned[i]
        short_breakout = close[i] < L4_aligned[i]
        
        long_entry = uptrend and long_breakout and volume_filter and volatility_filter[i]
        short_entry = downtrend and short_breakout and volume_filter and volatility_filter[i]
        
        # Exit conditions: Close below/above opposite Camarilla level (H3/L3 for exits)
        long_exit = close[i] < L3_aligned[i]
        short_exit = close[i] > H3_aligned[i]
        
        # Handle entries and exits
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif (position == 1 and long_exit) or (position == -1 and short_exit):
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

name = "12h_Camarilla_H4L4_Breakout_VolumeTrend_ATRFilter_v1"
timeframe = "12h"
leverage = 1.0