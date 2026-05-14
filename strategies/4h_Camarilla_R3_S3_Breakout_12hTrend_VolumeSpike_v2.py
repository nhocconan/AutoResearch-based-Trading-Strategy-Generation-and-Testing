#!/usr/bin/env python3
name = "4h_Camarilla_R3_S3_Breakout_12hTrend_VolumeSpike_v2"
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
    
    # 12h trend filter: EMA50
    df_12h = get_htf_data(prices, '12h')
    ema50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # 12h volume filter: volume > 1.5x 20-period average
    vol_ma_20_12h = pd.Series(df_12h['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_20_12h)
    
    # 12h price data for Camarilla levels
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Previous day's Camarilla levels (R3/S3)
    range_12h = high_12h - low_12h
    camarilla_h3 = close_12h + range_12h * 1.1 / 4
    camarilla_l3 = close_12h - range_12h * 1.1 / 4
    
    # Align to 4h
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_l3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if 12h trend or volume data not ready
        if np.isnan(ema50_12h_aligned[i]) or np.isnan(vol_ma_20_12h_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long conditions: price breaks above H3 with 12h uptrend and volume confirmation
            if (high[i] > camarilla_h3_aligned[i] and 
                close[i] > camarilla_h3_aligned[i] and
                close[i] > ema50_12h_aligned[i] and  # 12h uptrend
                volume[i] > vol_ma_20_12h_aligned[i]):  # volume spike
                signals[i] = 0.30
                position = 1
            # Short conditions: price breaks below L3 with 12h downtrend and volume confirmation
            elif (low[i] < camarilla_l3_aligned[i] and 
                  close[i] < camarilla_l3_aligned[i] and
                  close[i] < ema50_12h_aligned[i] and  # 12h downtrend
                  volume[i] > vol_ma_20_12h_aligned[i]):  # volume spike
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit long when price breaks below L3 or reverses against trend
            if (low[i] < camarilla_l3_aligned[i] or 
                close[i] < ema50_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit short when price breaks above H3 or reverses against trend
            if (high[i] > camarilla_h3_aligned[i] or 
                close[i] > ema50_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals