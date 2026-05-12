#!/usr/bin/env python3
name = "4h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike"
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
    
    # === 1D DATA FOR CAMARILLA PIVOTS AND TREND ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels from previous day
    # R3 = close + (high - low) * 1.1/2
    # S3 = close - (high - low) * 1.1/2
    range_1d = high_1d - low_1d
    camarilla_r3 = close_1d + range_1d * 1.1 / 2
    camarilla_s3 = close_1d - range_1d * 1.1 / 2
    
    # Align Camarilla levels to 4h timeframe
    camarilla_r3_4h = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_4h = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # 1D EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_4h = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # === VOLUME CONFIRMATION (20-period) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)  # Strong volume spike
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_r3_4h[i]) or np.isnan(camarilla_s3_4h[i]) or 
            np.isnan(ema34_1d_4h[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above R3 + uptrend + volume confirmation
            if (close[i] > camarilla_r3_4h[i] and 
                close[i] > ema34_1d_4h[i] and
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S3 + downtrend + volume confirmation
            elif (close[i] < camarilla_s3_4h[i] and 
                  close[i] < ema34_1d_4h[i] and
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Price breaks below S3 or trend breaks down
            if close[i] < camarilla_s3_4h[i] or close[i] < ema34_1d_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above R3 or trend breaks up
            if close[i] > camarilla_r3_4h[i] or close[i] > ema34_1d_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals