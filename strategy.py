#!/usr/bin/env python3
name = "12h_Camarilla_R1S1_Breakout_1wTrend_Volume"
timeframe = "12h"
leverage = 1.0

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
    
    # Get 1d data for Camarilla pivot calculation (from previous day)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day's OHLC
    # H, L, C from previous day
    H = df_1d['high'].values
    L = df_1d['low'].values
    C = df_1d['close'].values
    
    # Camarilla R1, S1 (most significant levels)
    # R1 = C + (H - L) * 1.1 / 12
    # S1 = C - (H - L) * 1.1 / 12
    R1 = C + (H - L) * 1.1 / 12
    S1 = C - (H - L) * 1.1 / 12
    
    # Align Camarilla levels to 12h timeframe (use previous day's levels)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # Get 1w data for trend filter (EMA34 on weekly close)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_up_1w = close_1w > ema34_1w
    trend_up_1w_aligned = align_htf_to_ltf(prices, df_1w, trend_up_1w)
    
    # Volume confirmation: current volume > 1.8x 30-period average
    vol_ma30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_filter = volume > 1.8 * vol_ma30
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Need enough data for volume MA
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or
            np.isnan(trend_up_1w_aligned[i]) or np.isnan(vol_ma30[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Price breaks above R1 + weekly uptrend + volume confirmation
            if close[i] > R1_aligned[i] and trend_up_1w_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S1 + weekly downtrend + volume confirmation
            elif close[i] < S1_aligned[i] and not trend_up_1w_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Price breaks below S1 OR weekly trend turns down
            if close[i] < S1_aligned[i] or not trend_up_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Price breaks above R1 OR weekly trend turns up
            if close[i] > R1_aligned[i] or trend_up_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals