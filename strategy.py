#!/usr/bin/env python3
name = "4h_Camarilla_R1_S1_Breakout_1wTrend_Volume"
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
    
    # 1w trend filter: EMA50
    df_1w = get_htf_data(prices, '1w')
    ema50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # 1d volume filter: volume > 1.5x 20-period average
    df_1d = get_htf_data(prices, '1d')
    vol_ma_20_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Daily Camarilla levels (based on previous day)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's Camarilla levels
    range_1d = high_1d - low_1d
    camarilla_h1 = close_1d + range_1d * 1.1 / 12
    camarilla_l1 = close_1d - range_1d * 1.1 / 12
    camarilla_h2 = close_1d + range_1d * 1.1 / 6
    camarilla_l2 = close_1d - range_1d * 1.1 / 6
    
    # Align to 4h
    camarilla_h1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h1)
    camarilla_l1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l1)
    camarilla_h2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h2)
    camarilla_l2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l2)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if 1w trend or 1d volume data not ready
        if np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_ma_20_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long conditions: price breaks above H1 with 1w uptrend and volume confirmation
            if (high[i] > camarilla_h1_aligned[i] and 
                close[i] > camarilla_h1_aligned[i] and
                close[i] > ema50_1w_aligned[i] and  # 1w uptrend
                volume[i] > vol_ma_20_1d_aligned[i]):  # volume spike
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below L1 with 1w downtrend and volume confirmation
            elif (low[i] < camarilla_l1_aligned[i] and 
                  close[i] < camarilla_l1_aligned[i] and
                  close[i] < ema50_1w_aligned[i] and  # 1w downtrend
                  volume[i] > vol_ma_20_1d_aligned[i]):  # volume spike
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long when price breaks below L2 or reverses against trend
            if (low[i] < camarilla_l2_aligned[i] or 
                close[i] < ema50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short when price breaks above H2 or reverses against trend
            if (high[i] > camarilla_h2_aligned[i] or 
                close[i] > ema50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals