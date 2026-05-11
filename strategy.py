#!/usr/bin/env python3
name = "4h_Camarilla_R1S1_Breakout_1dTrend_Volume"
timeframe = "4h"
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
    
    # Get 1d data for trend filter (EMA34) and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # Calculate EMA34 on daily
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_up_1d = close_1d > ema34_1d
    trend_up_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_up_1d)
    
    # Calculate Camarilla levels from daily high, low, close
    H = df_1d['high'].values
    L = df_1d['low'].values
    C = df_1d['close'].values
    
    # Camarilla formulas
    R4 = C + ((H - L) * 1.5 / 2)
    R3 = C + ((H - L) * 1.25 / 2)
    R2 = C + ((H - L) * 1.1 / 2)
    R1 = C + ((H - L) * 1.05 / 2)
    S1 = C - ((H - L) * 1.05 / 2)
    S2 = C - ((H - L) * 1.1 / 2)
    S3 = C - ((H - L) * 1.25 / 2)
    S4 = C - ((H - L) * 1.5 / 2)
    
    # Align all Camarilla levels to 4h timeframe
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > 1.5 * vol_ma20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for EMA and volume MA
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or
            np.isnan(trend_up_1d_aligned[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Price > R1 + 1d uptrend + volume confirmation
            if close[i] > R1_aligned[i] and trend_up_1d_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price < S1 + 1d downtrend + volume confirmation
            elif close[i] < S1_aligned[i] and not trend_up_1d_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Price < S1 OR 1d trend turns down
            if close[i] < S1_aligned[i] or not trend_up_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Price > R1 OR 1d trend turns up
            if close[i] > R1_aligned[i] or trend_up_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals