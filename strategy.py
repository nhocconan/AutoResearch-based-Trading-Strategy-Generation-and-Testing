#!/usr/bin/env python3
"""
4h_1d_4w_Camarilla_Pivot_Volume_Trend_v1
Hypothesis: Camarilla pivot levels from 1d timeframe provide high-probability reversal zones.
Trade long at L3 support and short at H3 resistance with volume confirmation and 4h trend filter.
Use 1w EMA50 trend filter to avoid counter-trend trades. Designed for 20-40 trades/year by requiring
multiple confluence factors: price at Camarilla level (within 0.2%), volume spike (>1.5x average),
and trend alignment (price > 4h EMA20 for longs, price < 4h EMA20 for shorts).
Works in bull markets via buying dips at support and in bear markets via selling rallies at resistance.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_4w_Camarilla_Pivot_Volume_Trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h EMA20 for trend filter
    ema_20_4h = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Volume average (20 period) for spike detection
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Load 1d data ONCE for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: H4, H3, H2, H1, L1, L2, L3, L4
    # Based on previous day's range
    camarilla_h4 = close_1d + 1.5 * (high_1d - low_1d)
    camarilla_h3 = close_1d + 1.1 * (high_1d - low_1d)
    camarilla_h2 = close_1d + 0.7 * (high_1d - low_1d)
    camarilla_h1 = close_1d + 0.5 * (high_1d - low_1d)
    camarilla_l1 = close_1d - 0.5 * (high_1d - low_1d)
    camarilla_l2 = close_1d - 0.7 * (high_1d - low_1d)
    camarilla_l3 = close_1d - 1.1 * (high_1d - low_1d)
    camarilla_l4 = close_1d - 1.5 * (high_1d - low_1d)
    
    # Align Camarilla levels to 4h timeframe (use previous day's levels)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Load 1w data ONCE for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_20_4h[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Price at Camarilla H3 resistance (within 0.2%)
        at_h3 = abs(close[i] - camarilla_h3_aligned[i]) / camarilla_h3_aligned[i] <= 0.002
        
        # Price at Camarilla L3 support (within 0.2%)
        at_l3 = abs(close[i] - camarilla_l3_aligned[i]) / camarilla_l3_aligned[i] <= 0.002
        
        # Volume spike: current volume > 1.5x average
        volume_spike = volume[i] > vol_ma[i] * 1.5
        
        # Trend filters
        above_ema_20 = close[i] > ema_20_4h[i]
        below_ema_20 = close[i] < ema_20_4h[i]
        above_ema_50_1w = close[i] > ema_50_1w_aligned[i]
        below_ema_50_1w = close[i] < ema_50_1w_aligned[i]
        
        # Entry conditions
        long_entry = at_l3 and volume_spike and above_ema_20 and above_ema_50_1w
        short_entry = at_h3 and volume_spike and below_ema_20 and below_ema_50_1w
        
        # Exit conditions: price moves back to VWAP equivalent (using close)
        # Simple exit: reverse signal or price moves 0.5% away from level
        long_exit = close[i] > camarilla_l3_aligned[i] * 1.005 or at_h3
        short_exit = close[i] < camarilla_h3_aligned[i] * 0.995 or at_l3
        
        # Priority: entry > exit > hold
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
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals