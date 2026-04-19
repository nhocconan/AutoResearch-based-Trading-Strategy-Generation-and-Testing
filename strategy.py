#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian breakout (20) + 1d volume spike (>2x) + 1d EMA trend filter
# Enter long when price breaks above 20-period high with volume confirmation and above 1d EMA50
# Enter short when price breaks below 20-period low with volume confirmation and below 1d EMA50
# Exit when price crosses back through the 10-period Donchian midpoint
# Uses discrete position sizing (0.25) to minimize fee churn
# Designed for 12h timeframe to target 15-30 trades/year, avoiding excessive turnover
name = "12h_Donchian20_1dVol_EMA50_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for volume and EMA filters
    df_1d = get_htf_data(prices, '1d')
    
    # 1d Volume average (20-period)
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 12h Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (high_20 + low_20) / 2  # 10-period midpoint for exit
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if np.isnan(high_20[i]) or np.isnan(low_20[i]) or np.isnan(vol_ma_1d_aligned[i]) or np.isnan(ema_50_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Volume filter: current 12h volume > 2x 1d average volume (scaled to 12h)
        # Approximate 12h volume from 1d: 1d volume / 2 (since 12h is half a day)
        vol_threshold = 2 * vol_ma_1d_aligned[i] / 2  # Adjust for timeframe difference
        volume_filter = volume[i] > vol_threshold if vol_threshold > 0 else False
        
        if position == 0:
            # Long entry: break above 20-period high + volume + above 1d EMA50
            if close[i] > high_20[i] and volume_filter and close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short entry: break below 20-period low + volume + below 1d EMA50
            elif close[i] < low_20[i] and volume_filter and close[i] < ema_50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below Donchian midpoint (10-period)
            if close[i] < donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above Donchian midpoint (10-period)
            if close[i] > donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals