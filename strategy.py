#!/usr/bin/env python3
name = "12h_Camarilla_R3S3_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d trend filter: EMA34
    df_1d = get_htf_data(prices, '1d')
    ema34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate Camarilla levels from previous 1d bar
    def camarilla_levels(high, low, close):
        range_val = high - low
        if range_val == 0:
            return (close, close, close, close, close, close)
        C = close + (range_val * 1.1 / 6)
        D = close + (range_val * 1.1 / 4)
        E = close + (range_val * 1.1 / 2)
        L = close - (range_val * 1.1 / 6)
        S = close - (range_val * 1.1 / 4)
        R = close - (range_val * 1.1 / 2)
        return (R, S, L, E, D, C)
    
    # Get previous 1d close for Camarilla calculation
    prev_1d_close = ema34_1d  # Use EMA as proxy for 1d close
    prev_1d_high = pd.Series(df_1d['high'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    prev_1d_low = pd.Series(df_1d['low'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    R, S, L, E, D, C = camarilla_levels(prev_1d_high, prev_1d_low, prev_1d_close)
    R_aligned = align_htf_to_ltf(prices, df_1d, R)
    S_aligned = align_htf_to_ltf(prices, df_1d, S)
    L_aligned = align_htf_to_ltf(prices, df_1d, L)
    E_aligned = align_htf_to_ltf(prices, df_1d, E)
    D_aligned = align_htf_to_ltf(prices, df_1d, D)
    C_aligned = align_htf_to_ltf(prices, df_1d, C)
    
    # Volume filter: volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # need enough data for EMA34 and volume MA
    
    for i in range(start_idx, n):
        # Skip if 1d trend data not ready
        if np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above S3 (S) with volume confirmation + 1d uptrend
            if (close[i] > S_aligned[i] and 
                volume[i] > 1.5 * vol_ma[i] and 
                close[i] > ema34_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below R3 (R) with volume confirmation + 1d downtrend
            elif (close[i] < R_aligned[i] and 
                  volume[i] > 1.5 * vol_ma[i] and 
                  close[i] < ema34_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long when price closes below EMA34 (1d trend) or below D (Camarilla)
            if close[i] < ema34_1d_aligned[i] or close[i] < D_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short when price closes above EMA34 (1d trend) or above C (Camarilla)
            if close[i] > ema34_1d_aligned[i] or close[i] > C_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals