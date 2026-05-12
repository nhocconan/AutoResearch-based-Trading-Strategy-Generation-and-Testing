#!/usr/bin/env python3
name = "4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeS"
timeframe = "4h"
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
    
    # 1d trend filter: EMA34
    df_1d = get_htf_data(prices, '1d')
    ema34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # 1d volume average (20-period)
    vol20_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol20_1d)
    
    # Previous day's OHLC for Camarilla levels
    prev_close = df_1d['close'].values
    prev_high = df_1d['high'].values
    prev_low = df_1d['low'].values
    
    # Camarilla levels for today (based on previous day)
    # R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    R1 = prev_close + (prev_high - prev_low) * 1.1 / 12
    S1 = prev_close - (prev_high - prev_low) * 1.1 / 12
    
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # need enough data for 1d EMA34 and volume
    
    for i in range(start_idx, n):
        # Skip if 1d trend or volume data not ready
        if np.isnan(ema34_1d_aligned[i]) or np.isnan(vol20_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        vol_ratio = volume[i] / vol20_1d_aligned[i] if vol20_1d_aligned[i] > 0 else 0
        
        if position == 0:
            # Long: Close > R1 + volume spike + 1d uptrend
            if (close[i] > R1_aligned[i] and 
                vol_ratio > 1.5 and 
                close[i] > ema34_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Close < S1 + volume spike + 1d downtrend
            elif (close[i] < S1_aligned[i] and 
                  vol_ratio > 1.5 and 
                  close[i] < ema34_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long when close < S1 or trend turns down
            if (close[i] < S1_aligned[i] or 
                close[i] < ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short when close > R1 or trend turns up
            if (close[i] > R1_aligned[i] or 
                close[i] > ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals