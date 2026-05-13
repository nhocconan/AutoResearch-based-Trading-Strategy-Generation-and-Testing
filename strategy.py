#!/usr/bin/env python3
name = "6h_Camarilla_R3S3_Breakout_1dTrend_Volume"
timeframe = "6h"
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
    
    # Calculate Camarilla levels from previous 1d bar
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    prev_1d_high = df_1d['high'].shift(1).values
    prev_1d_low = df_1d['low'].shift(1).values
    prev_1d_close = df_1d['close'].shift(1).values
    
    # Camarilla R3, S3, R4, S4
    R3 = prev_1d_close + (prev_1d_high - prev_1d_low) * 1.1 / 4
    S3 = prev_1d_close - (prev_1d_high - prev_1d_low) * 1.1 / 4
    R4 = prev_1d_close + (prev_1d_high - prev_1d_low) * 1.1 / 2
    S4 = prev_1d_close - (prev_1d_high - prev_1d_low) * 1.1 / 2
    
    # Align Camarilla levels to 6h timeframe
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    R4_aligned = align_htf_to_ltf(prices, df_1d, R4)
    S4_aligned = align_htf_to_ltf(prices, df_1d, S4)
    
    # 1d trend filter: EMA(34)
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume filter: current volume > 1.5 x 20-period average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        if (np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or 
            np.isnan(R4_aligned[i]) or np.isnan(S4_aligned[i]) or
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_condition = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # LONG: Break above R4 with bullish 1d trend and volume
            if close[i] > R4_aligned[i] and close[i] > ema34_1d_aligned[i] and vol_condition:
                signals[i] = 0.25
                position = 1
            # SHORT: Break below S4 with bearish 1d trend and volume
            elif close[i] < S4_aligned[i] and close[i] < ema34_1d_aligned[i] and vol_condition:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price re-enters below R3 or trend reversal
            if close[i] < R3_aligned[i] or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price re-enters above S3 or trend reversal
            if close[i] > S3_aligned[i] or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals