#!/usr/bin/env python3
name = "4h_Camarilla_R1_S1_Breakout_1dTrend_Volume"
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
    
    # 1d trend: EMA34
    df_1d = get_htf_data(prices, '1d')
    ema34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # 1d volume filter: volume > 1.5x 20-period average
    vol_ma_20_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Daily Camarilla levels (based on previous day)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's Camarilla levels (R1/S1)
    range_1d = high_1d - low_1d
    camarilla_r1 = close_1d + range_1d * 1.1 / 12
    camarilla_s1 = close_1d - range_1d * 1.1 / 12
    
    # Align to 4h
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if 1d trend or volume data not ready
        if np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma_20_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long conditions: price breaks above R1 with 1d uptrend and volume confirmation
            if (high[i] > camarilla_r1_aligned[i] and 
                close[i] > camarilla_r1_aligned[i] and
                close[i] > ema34_1d_aligned[i] and  # 1d uptrend
                volume[i] > vol_ma_20_1d_aligned[i]):  # volume spike
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below S1 with 1d downtrend and volume confirmation
            elif (low[i] < camarilla_s1_aligned[i] and 
                  close[i] < camarilla_s1_aligned[i] and
                  close[i] < ema34_1d_aligned[i] and  # 1d downtrend
                  volume[i] > vol_ma_20_1d_aligned[i]):  # volume spike
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long when price breaks below S1 or reverses against trend
            if (low[i] < camarilla_s1_aligned[i] or 
                close[i] < ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short when price breaks above R1 or reverses against trend
            if (high[i] > camarilla_r1_aligned[i] or 
                close[i] > ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals