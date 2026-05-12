#!/usr/bin/env python3
name = "12h_Camarilla_R3_S3_Breakout_1wTrend_VolumeSpike_v1"
timeframe = "12h"
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
    
    # 1w trend filter: EMA34
    df_1w = get_htf_data(prices, '1w')
    ema34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # 1w volume filter: volume > 1.5x 20-period average
    vol_ma_20_1w = pd.Series(df_1w['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_20_1w)
    
    # 1w price data for Camarilla levels
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Previous week's Camarilla levels (R3/S3)
    range_1w = high_1w - low_1w
    camarilla_h3 = close_1w + range_1w * 1.1 / 4
    camarilla_l3 = close_1w - range_1w * 1.1 / 4
    
    # Align to 12h
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_l3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if 1w trend or volume data not ready
        if np.isnan(ema34_1w_aligned[i]) or np.isnan(vol_ma_20_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long conditions: price breaks above H3 with 1w uptrend and volume confirmation
            if (high[i] > camarilla_h3_aligned[i] and 
                close[i] > camarilla_h3_aligned[i] and
                close[i] > ema34_1w_aligned[i] and  # 1w uptrend
                volume[i] > vol_ma_20_1w_aligned[i]):  # volume spike
                signals[i] = 0.30
                position = 1
            # Short conditions: price breaks below L3 with 1w downtrend and volume confirmation
            elif (low[i] < camarilla_l3_aligned[i] and 
                  close[i] < camarilla_l3_aligned[i] and
                  close[i] < ema34_1w_aligned[i] and  # 1w downtrend
                  volume[i] > vol_ma_20_1w_aligned[i]):  # volume spike
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit long when price breaks below L3 or reverses against trend
            if (low[i] < camarilla_l3_aligned[i] or 
                close[i] < ema34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit short when price breaks above H3 or reverses against trend
            if (high[i] > camarilla_h3_aligned[i] or 
                close[i] > ema34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals