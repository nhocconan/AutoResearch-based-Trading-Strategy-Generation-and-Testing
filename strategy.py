#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 6h Camarilla Pivot from 1d + Elder Ray (Bull/Bear Power) + Volume Confirmation
# Hypothesis: Camarilla levels act as strong support/resistance on 1d timeframe.
# Elder Ray confirms institutional buying/selling pressure. Volume ensures conviction.
# Works in bull by taking longs at S1/S2 with bullish power, shorts at R1/R2 with bearish power.
# Works in bear by fading at R3/S3 with reversal confirmation. Target: 15-30 trades/year.
name = "6h_camarilla_elderay_1d_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla and Elder Ray calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate daily OHLC for Camarilla (use previous day's values)
    o_1d = df_1d['open'].values
    h_1d = df_1d['high'].values
    l_1d = df_1d['low'].values
    c_1d = df_1d['close'].values
    v_1d = df_1d['volume'].values
    
    # Previous day's values (shift by 1 to avoid look-ahead)
    o_1d_prev = np.concatenate([[np.nan], o_1d[:-1]])
    h_1d_prev = np.concatenate([[np.nan], h_1d[:-1]])
    l_1d_prev = np.concatenate([[np.nan], l_1d[:-1]])
    c_1d_prev = np.concatenate([[np.nan], c_1d[:-1]])
    
    # Calculate Camarilla levels for previous day
    # Range = H - L
    range_1d = h_1d_prev - l_1d_prev
    # Camarilla multipliers
    camarilla_mult = np.array([1.0833, 1.1666, 1.2500, 1.5000])  # R1,S1,R2,S2,R3,S3,R4,S4
    
    # Calculate levels
    camarilla_levels = np.full((len(c_1d_prev), 8), np.nan)
    for i in range(len(c_1d_prev)):
        if not np.isnan(range_1d[i]) and range_1d[i] > 0:
            camarilla_levels[i, 0] = c_1d_prev[i] + range_1d[i] * 1.0833  # R1
            camarilla_levels[i, 1] = c_1d_prev[i] - range_1d[i] * 1.0833  # S1
            camarilla_levels[i, 2] = c_1d_prev[i] + range_1d[i] * 1.1666  # R2
            camarilla_levels[i, 3] = c_1d_prev[i] - range_1d[i] * 1.1666  # S2
            camarilla_levels[i, 4] = c_1d_prev[i] + range_1d[i] * 1.2500  # R3
            camarilla_levels[i, 5] = c_1d_prev[i] - range_1d[i] * 1.2500  # S3
            camarilla_levels[i, 6] = c_1d_prev[i] + range_1d[i] * 1.5000  # R4
            camarilla_levels[i, 7] = c_1d_prev[i] - range_1d[i] * 1.5000  # S4
    
    # Align Camarilla levels to 6h timeframe
    camarilla_aligned = []
    for i in range(8):
        camarilla_aligned.append(align_htf_to_ltf(prices, df_1d, camarilla_levels[:, i]))
    
    # Calculate Elder Ray (Bull/Bear Power) from 13-period EMA
    ema13 = pd.Series(c_1d).ewm(span=13, adjust=False).mean().values
    bull_power = h_1d - ema13  # High - EMA13
    bear_power = l_1d - ema13  # Low - EMA13
    
    # Align Elder Ray to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Volume confirmation: current 6h volume > 20-period average of 6h volume
    vol_ma_6h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(camarilla_aligned[0][i]) or np.isnan(bull_power_aligned[i]) or 
            np.isnan(bear_power_aligned[i]) or np.isnan(vol_ma_6h[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > vol_ma_6h[i]
        
        if position == 1:  # Long position
            # Exit: price crosses below S1 OR bear power turns negative
            if close[i] < camarilla_aligned[1][i] or bear_power_aligned[i] < 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit: price crosses above R1 OR bull power turns positive
            if close[i] > camarilla_aligned[0][i] or bull_power_aligned[i] > 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Enter long: price > S1 AND bull power > 0 AND volume confirmation
            if (close[i] > camarilla_aligned[1][i] and 
                bull_power_aligned[i] > 0 and vol_confirm):
                position = 1
                signals[i] = 0.25
            # Enter short: price < R1 AND bear power < 0 AND volume confirmation
            elif (close[i] < camarilla_aligned[0][i] and 
                  bear_power_aligned[i] < 0 and vol_confirm):
                position = -1
                signals[i] = -0.25
    
    return signals