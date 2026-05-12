#!/usr/bin/env python3
name = "12h_Camarilla_R3_S3_Breakout_1wTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 300:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1W DATA FOR TREND FILTER ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # 1W EMA40 for trend filter (longer-term trend)
    ema40_1w = pd.Series(close_1w).ewm(span=40, adjust=False, min_periods=40).mean().values
    ema40_1w_12h = align_htf_to_ltf(prices, df_1w, ema40_1w)
    
    # === 1D DATA FOR CAMARILLA PIVOTS ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels from previous day (R3/S3)
    prev_close = np.roll(close_1d, 1)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    range_hl = prev_high - prev_low
    
    r3 = prev_close + range_hl * 1.1 / 4
    s3 = prev_close - range_hl * 1.1 / 4
    
    # Align Camarilla levels to 12h timeframe
    r3_12h = align_htf_to_ltf(prices, df_1d, r3)
    s3_12h = align_htf_to_ltf(prices, df_1d, s3)
    
    # === VOLUME CONFIRMATION (30-period) ===
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(60, 30)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(r3_12h[i]) or np.isnan(s3_12h[i]) or np.isnan(ema40_1w_12h[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Break above R3 with volume, weekly uptrend
            if (close[i] > r3_12h[i] and 
                close[i] > ema40_1w_12h[i] and  # Weekly uptrend filter
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Break below S3 with volume, weekly downtrend
            elif (close[i] < s3_12h[i] and 
                  close[i] < ema40_1w_12h[i] and  # Weekly downtrend filter
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Weekly trend breaks down
            if close[i] < ema40_1w_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Weekly trend breaks up
            if close[i] > ema40_1w_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals