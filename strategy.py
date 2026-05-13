#!/usr/bin/env python3
"""
12h_1D_Camarilla_Pivot_S3_S4_Breakout_TrendFilter
Hypothesis: Buy near S3 and sell near S4 (or vice versa for shorts) on 12h timeframe using 1D Camarilla pivot levels, filtered by 1W EMA50 trend. Only enter on volume spikes (>1.5x 20-period average) to avoid chop. Designed for low trade frequency (12-37/year) to minimize fee drag while capturing institutional reversal zones. Works in bull/bear: Camarilla levels adapt to volatility, trend filter avoids counter-trend trades.
"""

name = "12h_1D_Camarilla_Pivot_S3_S4_Breakout_TrendFilter"
timeframe = "12h"
leverage = 1.0

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
    
    # Get daily data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day's OHLC
    # Camarilla: S3 = C - (H-L)*1.1/4, S4 = C - (H-L)*1.1/2
    #            R3 = C + (H-L)*1.1/4, R4 = C + (H-L)*1.1/2
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Shift by 1 to use previous day's data (no look-ahead)
    S3 = close_1d - (high_1d - low_1d) * 1.1 / 4
    S4 = close_1d - (high_1d - low_1d) * 1.1 / 2
    R3 = close_1d + (high_1d - low_1d) * 1.1 / 4
    R4 = close_1d + (high_1d - low_1d) * 1.1 / 2
    
    # Align to 12h timeframe (no extra delay for pivot levels)
    S3_12h = align_htf_to_ltf(prices, df_1d, S3)
    S4_12h = align_htf_to_ltf(prices, df_1d, S4)
    R3_12h = align_htf_to_ltf(prices, df_1d, R3)
    R4_12h = align_htf_to_ltf(prices, df_1d, R4)
    
    # Get weekly EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate volume average (20-period) for volume spike filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(S3_12h[i]) or np.isnan(S4_12h[i]) or 
            np.isnan(R3_12h[i]) or np.isnan(R4_12h[i]) or
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume spike condition: current volume > 1.5x 20-period average
        vol_spike = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # LONG: Price crosses above S3 with volume spike and above weekly EMA50 (uptrend)
            if close[i-1] <= S3_12h[i-1] and close[i] > S3_12h[i] and vol_spike and close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price crosses below R3 with volume spike and below weekly EMA50 (downtrend)
            elif close[i-1] >= R3_12h[i-1] and close[i] < R3_12h[i] and vol_spike and close[i] < ema_50_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below S4 or trend changes
            if close[i] < S4_12h[i] or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above R4 or trend changes
            if close[i] > R4_12h[i] or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals