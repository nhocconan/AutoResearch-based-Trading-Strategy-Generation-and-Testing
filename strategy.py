#!/usr/bin/env python3
name = "4h_Camarilla_R3_S3_Breakout_1dTrend_VolumeS"
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
    
    # Daily timeframe data for Camarilla levels and trend filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels from previous day
    # R3 = C + (H-L)*1.1/2, S3 = C - (H-L)*1.1/2
    camarilla_R3 = close_1d + (high_1d - low_1d) * 1.1 / 2
    camarilla_S3 = close_1d - (high_1d - low_1d) * 1.1 / 2
    
    # Align Camarilla levels to 4h timeframe (previous day's levels)
    camarilla_R3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R3)
    camarilla_S3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S3)
    
    # Daily trend filter: EMA34
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # ensure EMA34 has enough data
    
    for i in range(start_idx, n):
        # Skip if EMA data not ready
        if np.isnan(ema34_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R3 + above daily EMA34 + volume spike
            if (close[i] > camarilla_R3_aligned[i]) and (close[i] > ema34_1d_aligned[i]) and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 + below daily EMA34 + volume spike
            elif (close[i] < camarilla_S3_aligned[i]) and (close[i] < ema34_1d_aligned[i]) and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below S3
            if close[i] < camarilla_S3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above R3
            if close[i] > camarilla_R3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals