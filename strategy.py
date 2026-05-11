#!/usr/bin/env python3
name = "1h_Camarilla_R1_S1_Breakout_4hTrend"
timeframe = "1h"
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
    
    # 4h trend: EMA50
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # 1h Camarilla levels (based on previous day's OHLC)
    # Use 1d data to calculate Camarilla levels for 1h chart
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Pre-calculate daily Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla R1 and S1: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    rng = high_1d - low_1d
    r1_1d = close_1d + rng * 1.1 / 12
    s1_1d = close_1d - rng * 1.1 / 12
    
    # Align daily Camarilla levels to 1h timeframe
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # Volume filter: 1h volume > 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(50, 20)  # Wait for EMA50 and volume MA
    
    for i in range(start_idx, n):
        if np.isnan(ema_50_4h_aligned[i]) or np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R1 with volume, in 4h uptrend
            if close[i] > r1_1d_aligned[i] and volume[i] > vol_ma[i] and close[i] > ema_50_4h_aligned[i]:
                signals[i] = 0.20
                position = 1
            # Short: price breaks below S1 with volume, in 4h downtrend
            elif close[i] < s1_1d_aligned[i] and volume[i] > vol_ma[i] and close[i] < ema_50_4h_aligned[i]:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price closes below S1 or 4h trend turns down
            if close[i] < s1_1d_aligned[i] or close[i] < ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price closes above R1 or 4h trend turns up
            if close[i] > r1_1d_aligned[i] or close[i] > ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals