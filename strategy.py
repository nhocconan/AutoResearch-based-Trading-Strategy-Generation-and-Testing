#!/usr/bin/env python3
"""
4h_1d_camarilla_breakout_v27
Uses Camarilla pivot levels from daily timeframe to identify support/resistance.
Enters long when price breaks above R4 with volume and volatility confirmation.
Enters short when price breaks below S4 with volume and volatility confirmation.
Exits when price returns to the daily pivot point (PP).
Uses 4h timeframe with daily Camarilla levels for structure.
Target: 20-40 trades/year to minimize fee drag while capturing meaningful breaks.
"""

name = "4h_1d_camarilla_breakout_v27"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given daily OHLC"""
    range_val = high - low
    pp = (high + low + close) / 3
    r4 = pp + (range_val * 1.1 / 2)
    r3 = pp + (range_val * 1.1 / 4)
    r2 = pp + (range_val * 1.1 / 6)
    r1 = pp + (range_val * 1.1 / 12)
    s1 = pp - (range_val * 1.1 / 12)
    s2 = pp - (range_val * 1.1 / 6)
    s3 = pp - (range_val * 1.1 / 4)
    s4 = pp - (range_val * 1.1 / 2)
    return pp, r1, r2, r3, r4, s1, s2, s3, s4

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Arrays to store daily levels
    pp_array = np.full_like(close_1d, np.nan)
    r4_array = np.full_like(close_1d, np.nan)
    s4_array = np.full_like(close_1d, np.nan)
    
    # Calculate for each day
    for i in range(len(df_1d)):
        pp, r1, r2, r3, r4, s1, s2, s3, s4 = calculate_camarilla(
            high_1d[i], low_1d[i], close_1d[i]
        )
        pp_array[i] = pp
        r4_array[i] = r4
        s4_array[i] = s4
    
    # Align daily levels to 4h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp_array)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4_array)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4_array)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    # Volatility filter: ATR(20) > 0.5 * ATR(50) to avoid low volatility periods
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], 
                         np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_20 = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    atr_50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    vol_filter = atr_20 > (0.5 * atr_50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(pp_aligned[i]) or np.isnan(r4_aligned[i]) or 
            np.isnan(s4_aligned[i]) or np.isnan(vol_confirm[i]) or 
            np.isnan(vol_filter[i])):
            signals[i] = 0.0
            continue
        
        # Long entry: price breaks above R4 with volume and volatility
        if (close[i] > r4_aligned[i] and vol_confirm[i] and vol_filter[i] and 
            position != 1):
            position = 1
            signals[i] = 0.25
        # Short entry: price breaks below S4 with volume and volatility
        elif (close[i] < s4_aligned[i] and vol_confirm[i] and vol_filter[i] and 
              position != -1):
            position = -1
            signals[i] = -0.25
        # Exit: price returns to daily pivot point (mean reversion to mean)
        elif position == 1 and close[i] <= pp_aligned[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] >= pp_aligned[i]:
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