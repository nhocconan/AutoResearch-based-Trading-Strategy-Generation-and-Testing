#!/usr/bin/env python3
# 6h_Angle_of_Descent_Trend_Follower
# Hypothesis: Uses 1d ADX and 6h price acceleration to capture trend continuation after pullbacks.
# In uptrend (1d ADX > 25), go long when 6h price accelerates upward (price > EMA8 and EMA8 > EMA21).
# In downtrend (1d ADX > 25), go short when price accelerates downward (price < EMA8 and EMA8 < EMA21).
# Uses volume confirmation to avoid false breaks. Designed for low trade frequency in both bull and bear markets.

name = "6h_Angle_of_Descent_Trend_Follower"
timeframe = "6h"
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
    
    # 1d data for ADX trend strength
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX(14)
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]
    tr2[0] = np.abs(high_1d[0] - close_1d[0])
    tr3[0] = np.abs(low_1d[0] - close_1d[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    plus_dm = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    minus_dm = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    plus_dm[0] = 0
    minus_dm[0] = 0
    
    tr_ma = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    plus_dm_ma = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    minus_dm_ma = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    plus_di = 100 * plus_dm_ma / tr_ma
    minus_di = 100 * minus_dm_ma / tr_ma
    dx = np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100
    dx = np.where((plus_di + minus_di) == 0, 0, dx)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Align ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # 6h EMAs for acceleration
    ema8 = pd.Series(close).ewm(span=8, adjust=False, min_periods=8).mean().values
    ema21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Volume confirmation: current volume > 1.3 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.3 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        if (np.isnan(adx_aligned[i]) or np.isnan(ema8[i]) or np.isnan(ema21[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: 1d uptrend (ADX > 25) + price acceleration up + volume
            if (adx_aligned[i] > 25 and 
                close[i] > ema8[i] and 
                ema8[i] > ema21[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: 1d downtrend (ADX > 25) + price acceleration down + volume
            elif (adx_aligned[i] > 25 and 
                  close[i] < ema8[i] and 
                  ema8[i] < ema21[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: trend acceleration lost or ADX weak
            if not (close[i] > ema8[i] and ema8[i] > ema21[i]) or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: trend acceleration lost or ADX weak
            if not (close[i] < ema8[i] and ema8[i] < ema21[i]) or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals