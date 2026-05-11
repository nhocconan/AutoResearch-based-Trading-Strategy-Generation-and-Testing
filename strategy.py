#!/usr/bin/env python3
name = "1D_Camarilla_R3S3_Breakout_1WTrend_Volume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly data for trend and Camarilla
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Weekly trend: 8-period EMA
    ema_8_1w = pd.Series(close_1w).ewm(span=8, adjust=False, min_periods=8).mean().values
    
    # Camarilla levels from previous week
    # H, L, C from previous weekly bar
    H = np.roll(high_1w, 1)
    L = np.roll(low_1w, 1)
    C = np.roll(close_1w, 1)
    H[0] = high_1w[0]  # avoid NaN on first bar
    L[0] = low_1w[0]
    C[0] = close_1w[0]
    
    # Camarilla R3, S3, R4, S4
    R3 = C + (H - L) * 1.1 / 4
    S3 = C - (H - L) * 1.1 / 4
    R4 = C + (H - L) * 1.1 / 2
    S4 = C - (H - L) * 1.1 / 2
    
    # Align weekly data to daily
    ema_8_aligned = align_htf_to_ltf(prices, df_1w, ema_8_1w)
    R3_aligned = align_htf_to_ltf(prices, df_1w, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1w, S3)
    R4_aligned = align_htf_to_ltf(prices, df_1w, R4)
    S4_aligned = align_htf_to_ltf(prices, df_1w, S4)
    
    # Volume spike (20-day average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(20, n):
        if np.isnan(ema_8_aligned[i]) or np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or np.isnan(R4_aligned[i]) or np.isnan(S4_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: break above R3 with volume spike and weekly uptrend
            if close[i] > R3_aligned[i] and vol_spike[i] and close[i] > ema_8_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below S3 with volume spike and weekly downtrend
            elif close[i] < S3_aligned[i] and vol_spike[i] and close[i] < ema_8_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: break below S4 or weekly trend turns down
            if close[i] < S4_aligned[i] or close[i] < ema_8_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: break above R4 or weekly trend turns up
            if close[i] > R4_aligned[i] or close[i] > ema_8_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals