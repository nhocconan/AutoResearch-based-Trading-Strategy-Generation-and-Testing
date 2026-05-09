#!/usr/bin/env python3
# 4h_Camarilla_R1_S1_Breakout_12hEMA50_Trend_VolumeS
# Hypothesis: Camarilla R1/S1 breakout with 12h EMA50 trend filter and volume spike works in both bull and bear markets by capturing institutional breakout directions while avoiding counter-trend trades. Volume confirmation reduces false breaks. Target: 20-50 trades/year.

name = "4h_Camarilla_R1_S1_Breakout_12hEMA50_Trend_VolumeS"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate 20-period average volume for spike detection
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_ma_20[i]) or vol_ma_20[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate Camarilla levels for today using previous day's OHLC
        if i >= 6:  # Need at least 6 periods back for previous day (4h * 6 = 24h)
            prev_day_high = high[i-6]
            prev_day_low = low[i-6]
            prev_day_close = close[i-6]
            
            # Camarilla calculations
            range_val = prev_day_high - prev_day_low
            if range_val <= 0:
                if position != 0:
                    signals[i] = 0.0
                    position = 0
                continue
                
            # R1 and S1 levels
            r1 = prev_day_close + (range_val * 1.1 / 12)
            s1 = prev_day_close - (range_val * 1.1 / 12)
        else:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike condition (2x average)
        volume_spike = volume[i] > (2.0 * vol_ma_20[i])
        
        if position == 0:
            # Enter long: price breaks above R1 with uptrend filter and volume spike
            if close[i] > r1 and close[i] > ema_50_12h_aligned[i] and volume_spike:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below S1 with downtrend filter and volume spike
            elif close[i] < s1 and close[i] < ema_50_12h_aligned[i] and volume_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below S1 or trend changes
            if close[i] < s1 or close[i] < ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above R1 or trend changes
            if close[i] > r1 or close[i] > ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals