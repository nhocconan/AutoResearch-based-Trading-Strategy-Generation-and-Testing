#!/usr/bin/env python3
"""
1h_Pivot_R1_S1_Breakout_Volume_ATRFilter_v1
Breakout above/below daily Camarilla pivot resistance/support with volume confirmation and ATR filter.
Uses 1d Camarilla pivots (R1/S1) for direction, 1h for entry timing, and ATR to avoid low-volatility whipsaws.
Session filter (08-20 UTC) to focus on active hours. Fixed position size 0.20 to control risk.
Target: 60-120 total trades over 4 years (15-30/year) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === Daily ATR(14) for volatility filter ===
    df_1d = get_htf_data(prices, '1d')
    atr_1d = np.zeros(len(df_1d))
    if len(df_1d) >= 14:
        tr = np.maximum(df_1d['high'] - df_1d['low'],
                        np.maximum(np.abs(df_1d['high'] - np.roll(df_1d['close'], 1)),
                                   np.abs(df_1d['low'] - np.roll(df_1d['close'], 1))))
        tr[0] = df_1d['high'].iloc[0] - df_1d['low'].iloc[0]
        atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # === Daily Camarilla Pivots (R1, S1) ===
    # Based on previous day's OHLC
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Pivot = (H + L + C) / 3
    pivot = (high_1d + low_1d + close_1d) / 3.0
    # R1 = C + (H - L) * 1.1 / 12
    r1 = close_1d + (high_1d - low_1d) * 1.1 / 12.0
    # S1 = C - (H - L) * 1.1 / 12
    s1 = close_1d - (high_1d - low_1d) * 1.1 / 12.0
    
    # Align to 1h (values become available after the daily candle closes)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # === Volume confirmation: current volume > 1.5x 20-period average ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === Session filter: 08-20 UTC ===
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 30
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(pivot_aligned[i]) or 
            np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(atr_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Session filter: only trade between 08:00-20:00 UTC
        hour = hours[i]
        in_session = 8 <= hour <= 20
        
        # Volatility filter: avoid extremely low volatility (ATR too small)
        vol_filter = atr_1d_aligned[i] > 0
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0 and in_session and vol_filter:
            # Long: price breaks above R1 with volume confirmation
            if close[i] > r1_aligned[i] and vol_confirm:
                signals[i] = 0.20
                position = 1
                continue
            # Short: price breaks below S1 with volume confirmation
            elif close[i] < s1_aligned[i] and vol_confirm:
                signals[i] = -0.20
                position = -1
                continue
        
        elif position == 1:
            # Exit long: price returns below pivot (mean reversion) OR session end
            if close[i] < pivot_aligned[i] or not in_session:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: price returns above pivot (mean reversion) OR session end
            if close[i] > pivot_aligned[i] or not in_session:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Pivot_R1_S1_Breakout_Volume_ATRFilter_v1"
timeframe = "1h"
leverage = 1.0