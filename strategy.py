#!/usr/bin/env python3
# 1h_4h1d_donchian_breakout_volume_v1
# Hypothesis: Trade Donchian breakouts on 1h with volume confirmation, filtered by 4h/1d trend.
# Long when price breaks above 20-period 1h Donchian high with volume surge and 4h/1d uptrend.
# Short when price breaks below 20-period 1h Donchian low with volume surge and 4h/1d downtrend.
# Designed for 1h timeframe to target 15-37 trades/year (60-150 total over 4 years).
# Uses 4h/1d trend filters to avoid counter-trend trades in both bull and bear markets.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4h1d_donchian_breakout_volume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    # 4h EMA50 for trend
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    # 1d EMA50 for trend
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # 1h Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 50  # Ensure EMA50 is ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema50_4h_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(high_20[i]) or np.isnan(low_20[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        # Volume surge condition
        vol_surge = volume[i] > 1.5 * vol_ma_20[i] if vol_ma_20[i] > 0 else False
        
        # Trend filter: both 4h and 1d EMA50 must agree
        uptrend_4h = close[i] > ema50_4h_aligned[i]
        uptrend_1d = close[i] > ema50_1d_aligned[i]
        downtrend_4h = close[i] < ema50_4h_aligned[i]
        downtrend_1d = close[i] < ema50_1d_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price breaks below 1h Donchian low
            if close[i] < low_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: price breaks above 1h Donchian high
            if close[i] > high_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat, look for entry
            # Long entry: price breaks above 1h Donchian high with volume surge and uptrend
            if (close[i] > high_20[i] and vol_surge and 
                uptrend_4h and uptrend_1d):
                position = 1
                signals[i] = 0.20
            # Short entry: price breaks below 1h Donchian low with volume surge and downtrend
            elif (close[i] < low_20[i] and vol_surge and 
                  downtrend_4h and downtrend_1d):
                position = -1
                signals[i] = -0.20
    
    return signals