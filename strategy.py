#!/usr/bin/env python3
name = "1h_Camarilla_R1_S1_Breakout_4hTrend_1dVolume"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 4h 14-period EMA for trend
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema_14_4h = pd.Series(close_4h).ewm(span=14, adjust=False).mean().values
    ema_14_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_14_4h)
    
    # Calculate 1d average volume (20 periods)
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    vol_avg_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)
    
    # Calculate 1h hourly close and volume for session filter
    hours = prices.index.hour
    
    # Calculate 1h Camarilla levels from 4h data
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    range_4h = high_4h - low_4h
    r1_4h = close_4h + range_4h * 1.1 / 12
    s1_4h = close_4h - range_4h * 1.1 / 12
    r1_4h_aligned = align_htf_to_ltf(prices, df_4h, r1_4h)
    s1_4h_aligned = align_htf_to_ltf(prices, df_4h, s1_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        if np.isnan(ema_14_4h_aligned[i]) or np.isnan(vol_avg_20_1d_aligned[i]) or np.isnan(r1_4h_aligned[i]) or np.isnan(s1_4h_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above 4h R1 with volume confirmation and 4h uptrend
            if close[i] > r1_4h_aligned[i] and volume[i] > 1.5 * vol_avg_20_1d_aligned[i] and close[i] > ema_14_4h_aligned[i]:
                signals[i] = 0.20
                position = 1
            # Short: Price breaks below 4h S1 with volume confirmation and 4h downtrend
            elif close[i] < s1_4h_aligned[i] and volume[i] > 1.5 * vol_avg_20_1d_aligned[i] and close[i] < ema_14_4h_aligned[i]:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit: Price re-enters the 4h H-L range (S1 to R1)
            if close[i] < r1_4h_aligned[i] and close[i] > s1_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit: Price re-enters the 4h H-L range (S1 to R1)
            if close[i] < r1_4h_aligned[i] and close[i] > s1_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals