#!/usr/bin/env python3
name = "4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for Camarilla levels, trend, and volatility
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate previous day's range for Camarilla levels
    prev_close = np.roll(close_1d, 1)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close[0] = close_1d[0]  # first day uses same day
    prev_high[0] = high_1d[0]
    prev_low[0] = low_1d[0]
    
    range_1d = prev_high - prev_low
    
    # Camarilla levels (based on previous day)
    S1 = prev_close - (range_1d * 1.0 / 12)
    S2 = prev_close - (range_1d * 2.0 / 12)
    S3 = prev_close - (range_1d * 3.0 / 12)
    R1 = prev_close + (range_1d * 1.0 / 12)
    R2 = prev_close + (range_1d * 2.0 / 12)
    R3 = prev_close + (range_1d * 3.0 / 12)
    
    # 34-period EMA for 1d trend
    ema34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Volatility filter: ATR(14) for stop and regime
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume spike: 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 2.0)  # Require strong volume spike
    
    # Align all 1d indicators to 4h timeframe
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    S2_aligned = align_htf_to_ltf(prices, df_1d, S2)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    R2_aligned = align_htf_to_ltf(prices, df_1d, R2)
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34)
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20, 14)  # warmup for EMA34, vol MA, ATR
    
    for i in range(start_idx, n):
        if np.isnan(S1_aligned[i]) or np.isnan(R1_aligned[i]) or np.isnan(ema34_aligned[i]) or np.isnan(atr_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: break above R1 with volume spike and price above 1d EMA34 (uptrend)
            if close[i] > R1_aligned[i] and vol_spike[i] and close[i] > ema34_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below S1 with volume spike and price below 1d EMA34 (downtrend)
            elif close[i] < S1_aligned[i] and vol_spike[i] and close[i] < ema34_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long: exit on break below S1 or close below EMA34
            if close[i] < S1_aligned[i] or close[i] < ema34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short: exit on break above R1 or close above EMA34
            if close[i] > R1_aligned[i] or close[i] > ema34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals