#!/usr/bin/env python3
# 4h_1d_camarilla_pivot_volume_v1
# Strategy: 4h Camarilla pivot levels with daily trend filter and volume confirmation
# Timeframe: 4h
# Leverage: 1.0
# Hypothesis: Camarilla pivot levels provide high-probability reversal zones; daily EMA filter ensures alignment with higher-timeframe trend; volume confirmation avoids false signals. Designed for low trade frequency (<30/year) to minimize fee drag in BTC/ETH.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_camarilla_pivot_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop for Camarilla pivots and trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: H4, L4, H3, L3, H2, L2, H1, L1
    # Formula: R4 = C + ((H-L) * 1.1/2), R3 = C + ((H-L) * 1.1/4), etc.
    # We'll use L3 and H3 as primary entry/exit levels
    camarilla_h3 = close_1d + ((high_1d - low_1d) * 1.1 / 4)
    camarilla_l3 = close_1d - ((high_1d - low_1d) * 1.1 / 4)
    
    # Daily EMA(50) for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align daily data to 4h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 4h Volume confirmation: current volume > 1.3x 20-period average
    vol_series = pd.Series(volume)
    vol_avg_20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (1.3 * vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Price near Camarilla levels (within 0.1% tolerance)
        near_h3 = abs(close[i] - camarilla_h3_aligned[i]) / camarilla_h3_aligned[i] < 0.001
        near_l3 = abs(close[i] - camarilla_l3_aligned[i]) / camarilla_l3_aligned[i] < 0.001
        
        # Trend filter: price above/below daily EMA50
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Entry logic: Near Camarilla level + volume + trend alignment for reversal
        if near_l3 and vol_confirm[i] and uptrend and position != 1:
            # Near L3 in uptrend = long (bounce)
            position = 1
            signals[i] = 0.25
        elif near_h3 and vol_confirm[i] and downtrend and position != -1:
            # Near H3 in downtrend = short (rejection)
            position = -1
            signals[i] = -0.25
        # Exit: price moves back toward mean (middle of range)
        elif position == 1 and close[i] > (camarilla_h3_aligned[i] + camarilla_l3_aligned[i]) / 2:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] < (camarilla_h3_aligned[i] + camarilla_l3_aligned[i]) / 2:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals