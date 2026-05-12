#!/usr/bin/env python3
"""
6H_CAMARILLA_R3_S3_BREAKOUT_1D_TREND_VOLUME
Hypothesis: Camarilla R3/S3 breakout on 6h with 1d EMA34 trend filter and volume spike confirmation. R3/S3 levels represent stronger support/resistance than R1/S1, reducing false breakouts. Trend filter ensures alignment with daily momentum, while volume spike confirms institutional participation. Designed to work in both bull and bear markets by capturing significant breakouts with confirmation.
"""
name = "6H_CAMARILLA_R3_S3_BREAKOUT_1D_TREND_VOLUME"
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
    
    # 1d data for trend filter (EMA34)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # 1d data for Camarilla levels (using previous day's range)
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's range for Camarilla calculation
    ph = np.roll(high_1d, 1)  # previous day high
    pl = np.roll(low_1d, 1)   # previous day low
    pc = np.roll(close_1d, 1) # previous day close
    ph[0] = pl[0] = pc[0] = np.nan  # first day has no previous day
    
    # Camarilla levels: R3 = C + (H-L)*1.1/4, S3 = C - (H-L)*1.1/4
    r3 = pc + (ph - pl) * 1.1 / 4
    s3 = pc - (ph - pl) * 1.1 / 4
    
    # Align Camarilla levels to 6h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Volume spike: current 6h volume > 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=1).mean().values
    volume_spike = volume > 1.8 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Break above R3 with volume spike and price above 1d EMA34
            if (high[i] > r3_aligned[i] and 
                volume_spike[i] and 
                close[i] > ema34_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Break below S3 with volume spike and price below 1d EMA34
            elif (low[i] < s3_aligned[i] and 
                  volume_spike[i] and 
                  close[i] < ema34_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price falls back below S3 or below 1d EMA34
            if (close[i] < s3_aligned[i] or 
                close[i] < ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price rises back above R3 or above 1d EMA34
            if (close[i] > r3_aligned[i] or 
                close[i] > ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals