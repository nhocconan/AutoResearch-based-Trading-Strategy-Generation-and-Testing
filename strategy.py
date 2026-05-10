#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1wTrend_Volume
Hypothesis: Price reversal at weekly Camarilla R1/S1 levels confirmed by 1w trend and volume spike.
Works in bull/bear via weekly trend filter + avoids false signals in low volume.
Targets 12-37 trades/year by requiring confluence of weekly price level, trend, and volume.
"""

name = "12h_Camarilla_R1_S1_Breakout_1wTrend_Volume"
timeframe = "12h"
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
    
    # Get 1w data for trend and volume filters
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate 1w EMA34 for trend filter
    ema_34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate 1w average volume for volume filter
    vol_avg_1w = pd.Series(df_1w['volume']).rolling(window=20, min_periods=20).mean().values
    vol_avg_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_avg_1w)
    
    # Calculate Camarilla levels from previous 1w OHLC
    # Use previous week's OHLC, so shift by 1
    prev_close = np.concatenate([[df_1w['close'].iloc[0]], df_1w['close'].values[:-1]])
    prev_high = np.concatenate([[df_1w['high'].iloc[0]], df_1w['high'].values[:-1]])
    prev_low = np.concatenate([[df_1w['low'].iloc[0]], df_1w['low'].values[:-1]])
    
    camarilla_range = prev_high - prev_low
    r1 = prev_close + camarilla_range * 1.1 / 12
    s1 = prev_close - camarilla_range * 1.1 / 12
    
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need 1w EMA34 (34) and 1w vol avg (20)
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(vol_avg_1w_aligned[i]) or 
            np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Higher timeframe trend filter (1w)
        uptrend_1w = close[i] > ema_34_1w_aligned[i]
        downtrend_1w = close[i] < ema_34_1w_aligned[i]
        
        # Volume filter: current 12h volume > 1.5x average 1w volume (scaled)
        vol_12h = volume[i]
        # Scale 1w volume to 12h equivalent (1w = 14x 12h)
        vol_12h_equiv = vol_avg_1w_aligned[i] / 14.0
        volume_filter = vol_12h > vol_12h_equiv * 1.5
        
        if position == 0:
            # Long entry: price at S1 support + uptrend + volume participation
            if close[i] <= s1_aligned[i] and uptrend_1w and volume_filter:
                signals[i] = 0.25
                position = 1
            # Short entry: price at R1 resistance + downtrend + volume participation
            elif close[i] >= r1_aligned[i] and downtrend_1w and volume_filter:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price reaches R1 or trend breaks
            if close[i] >= r1_aligned[i] or not uptrend_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price reaches S1 or trend breaks
            if close[i] <= s1_aligned[i] or not downtrend_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals