# -*- coding: utf-8 -*-
# 4h_1d_1w_trend_breakout_volume_v1
# Hypothesis: Use 1d trend (price > SMA50) and 1w momentum (price > SMA200) as regime filter.
# Enter long when 4h price breaks above Donchian(20) high with volume confirmation (>1.5x avg volume).
# Enter short when 4h price breaks below Donchian(20) low with volume confirmation.
# Exit on opposite Donchian break or when trend/momentum filter fails.
# Designed for fewer trades (~20-50/year) to avoid fee drag, works in bull (trend up) and bear (trend down) regimes.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_1w_trend_breakout_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter (SMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d SMA50 for trend filter
    close_1d = df_1d['close'].values
    sma50_1d = pd.Series(close_1d).rolling(window=50, min_periods=50).mean().values
    sma50_1d_aligned = align_htf_to_ltf(prices, df_1d, sma50_1d)
    
    # Get 1w data for momentum filter (SMA200)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    
    # Calculate 1w SMA200 for momentum filter
    close_1w = df_1w['close'].values
    sma200_1w = pd.Series(close_1w).rolling(window=200, min_periods=200).mean().values
    sma200_1w_aligned = align_htf_to_ltf(prices, df_1w, sma200_1w)
    
    # Calculate Donchian channels on 4h (20-period)
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 1.5x average of last 20 periods
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > vol_ma * 1.5
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup (max of 200, 50, 20)
    start_idx = 200
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(sma50_1d_aligned[i]) or np.isnan(sma200_1w_aligned[i]) or
            np.isnan(high_max[i]) or np.isnan(low_min[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price breaks below Donchian low OR trend/momentum fails
            if close[i] < low_min[i] or close[i] < sma50_1d_aligned[i] or close[i] < sma200_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
                
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian high OR trend/momentum fails
            if close[i] > high_max[i] or close[i] > sma50_1d_aligned[i] or close[i] > sma200_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Long entry: price breaks above Donchian high with volume and trend/momentum OK
            if close[i] > high_max[i] and vol_confirm[i] and close[i] > sma50_1d_aligned[i] and close[i] > sma200_1w_aligned[i]:
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below Donchian low with volume and trend/momentum OK
            elif close[i] < low_min[i] and vol_confirm[i] and close[i] < sma50_1d_aligned[i] and close[i] < sma200_1w_aligned[i]:
                position = -1
                signals[i] = -0.25
    
    return signals