#!/usr/bin/env python3
# 1h_4h1d_Camarilla_R1S1_TrendFollow_VolumeFilter
# Hypothesis: Daily and 4h Camarilla R1/S1 breakouts on 1h timeframe with 4h/1d EMA trend filter and volume confirmation.
# Uses 4h/1d EMA for trend and volume spike to avoid false breakouts.
# Target: 15-37 trades/year per symbol by using 1h only for entry timing, with 4h/1d for direction.
# Session filter (08-20 UTC) reduces noise trades. Position size fixed at 0.20 to manage drawdown.

name = "1h_4h1d_Camarilla_R1S1_TrendFollow_VolumeFilter"
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
    
    # Get 4h and 1d data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 35 or len(df_1d) < 35:
        return np.zeros(n)
    
    # Calculate 4h pivot points
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    pivot_4h = (high_4h + low_4h + close_4h) / 3.0
    r1_4h = pivot_4h + (high_4h - low_4h) * 1.1 / 12
    s1_4h = pivot_4h - (high_4h - low_4h) * 1.1 / 12
    
    # Calculate 4h EMA34 for trend filter
    ema34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate 1d pivot points
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    r1_1d = pivot_1d + (high_1d - low_1d) * 1.1 / 12
    s1_1d = pivot_1d - (high_1d - low_1d) * 1.1 / 12
    
    # Calculate 1d EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate volume average for spike detection (24 * 4h = 4 days)
    vol_ma_4h = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    # Align 4h and 1d indicators to 1h timeframe
    r1_4h_aligned = align_htf_to_ltf(prices, df_4h, r1_4h)
    s1_4h_aligned = align_htf_to_ltf(prices, df_4h, s1_4h)
    ema34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema34_4h)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    vol_ma_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_4h)
    
    # Precompute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r1_4h_aligned[i]) or np.isnan(s1_4h_aligned[i]) or 
            np.isnan(ema34_4h_aligned[i]) or np.isnan(r1_1d_aligned[i]) or 
            np.isnan(s1_1d_aligned[i]) or np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(vol_ma_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            signals[i] = 0.0
            continue
        
        # Volume spike condition: current volume > 1.5 * 4h average volume
        volume_spike = volume[i] > 1.5 * vol_ma_4h_aligned[i]
        
        if position == 0:
            # Long: price > 4h EMA34 AND 1d EMA34 (uptrend) and breaks above R1_4h OR R1_1d with volume
            if (close[i] > ema34_4h_aligned[i] and close[i] > ema34_1d_aligned[i] and 
                (close[i] > r1_4h_aligned[i] or close[i] > r1_1d_aligned[i]) and volume_spike):
                signals[i] = 0.20
                position = 1
            # Short: price < 4h EMA34 AND 1d EMA34 (downtrend) and breaks below S1_4h OR S1_1d with volume
            elif (close[i] < ema34_4h_aligned[i] and close[i] < ema34_1d_aligned[i] and 
                  (close[i] < s1_4h_aligned[i] or close[i] < s1_1d_aligned[i]) and volume_spike):
                signals[i] = -0.20
                position = -1
                
        elif position == 1:
            # Long: exit if price breaks below S1_4h OR S1_1d (reversal) or trend changes
            if (close[i] < s1_4h_aligned[i] or close[i] < s1_1d_aligned[i] or 
                close[i] < ema34_4h_aligned[i] or close[i] < ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
                
        elif position == -1:
            # Short: exit if price breaks above R1_4h OR R1_1d (reversal) or trend changes
            if (close[i] > r1_4h_aligned[i] or close[i] > r1_1d_aligned[i] or 
                close[i] > ema34_4h_aligned[i] or close[i] > ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals