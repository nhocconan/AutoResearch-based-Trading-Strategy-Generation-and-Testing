#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeS
Hypothesis: Price reversal at Camarilla R1/S1 levels confirmed by 1d trend and volume spike.
Works in bull/bear via trend filter + avoids false signals in low volume.
Targets 20-40 trades/year by requiring confluence of price level, trend, and volume.
"""

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeS"
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
    
    # Get 1d data for trend and volume filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1d average volume for volume filter
    vol_avg_1d = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    vol_avg_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    # Calculate Camarilla levels from previous 1d OHLC
    # Camarilla: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    # We use previous day's OHLC, so shift by 1
    prev_close = np.concatenate([[df_1d['close'].iloc[0]], df_1d['close'].values[:-1]])
    prev_high = np.concatenate([[df_1d['high'].iloc[0]], df_1d['high'].values[:-1]])
    prev_low = np.concatenate([[df_1d['low'].iloc[0]], df_1d['low'].values[:-1]])
    
    camarilla_range = prev_high - prev_low
    r1 = prev_close + camarilla_range * 1.1 / 12
    s1 = prev_close - camarilla_range * 1.1 / 12
    
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need 1d EMA34 (34) and 1d vol avg (20)
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(vol_avg_1d_aligned[i]) or 
            np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Higher timeframe trend filter (1d)
        uptrend_1d = close[i] > ema_34_1d_aligned[i]
        downtrend_1d = close[i] < ema_34_1d_aligned[i]
        
        # Volume filter: current 4h volume > 1.5x average 1d volume (scaled)
        vol_4h = volume[i]
        # Scale 1d volume to 4h equivalent (1d = 6x 4h)
        vol_4h_equiv = vol_avg_1d_aligned[i] / 6.0
        volume_filter = vol_4h > vol_4h_equiv * 1.5
        
        if position == 0:
            # Long entry: price at S1 support + uptrend + volume participation
            if close[i] <= s1_aligned[i] and uptrend_1d and volume_filter:
                signals[i] = 0.25
                position = 1
            # Short entry: price at R1 resistance + downtrend + volume participation
            elif close[i] >= r1_aligned[i] and downtrend_1d and volume_filter:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price reaches R1 or trend breaks
            if close[i] >= r1_aligned[i] or not uptrend_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price reaches S1 or trend breaks
            if close[i] <= s1_aligned[i] or not downtrend_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals