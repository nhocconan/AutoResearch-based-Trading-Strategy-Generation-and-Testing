#!/usr/bin/env python3
name = "4h_Camarilla_R1_S1_Breakout_12hEMA50_Trend_VolumeS"
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
    
    # === 12H DATA FOR TREND FILTER ===
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # EMA50 on 12h for trend filter
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # === 1D DATA FOR CAMARILLA PIVOTS ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla pivot calculation (based on previous day)
    # Typical price = (H + L + C) / 3
    typical_price_1d = (high_1d + low_1d + close_1d) / 3.0
    # Pivot point = typical price
    pivot_1d = typical_price_1d
    # Range = high - low
    range_1d = high_1d - low_1d
    
    # Resistance levels: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    r1_1d = close_1d + (range_1d * 1.1 / 12)
    s1_1d = close_1d - (range_1d * 1.1 / 12)
    
    # Align Camarilla levels to 4h timeframe
    r1_4h = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_4h = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # === VOLUME CONFIRMATION (20-period) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 50, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50_12h_aligned[i]) or np.isnan(r1_4h[i]) or np.isnan(s1_4h[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above R1 + above 12h EMA50 + volume spike
            if (close[i] > r1_4h[i] and 
                close[i] > ema50_12h_aligned[i] and
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S1 + below 12h EMA50 + volume spike
            elif (close[i] < s1_4h[i] and 
                  close[i] < ema50_12h_aligned[i] and
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Price breaks below S1 OR below 12h EMA50
            if close[i] < s1_4h[i] or close[i] < ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above R1 OR above 12h EMA50
            if close[i] > r1_4h[i] or close[i] > ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals