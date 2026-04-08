#!/usr/bin/env python3
# 4h_12h_breakout_volume_ma_v1
# Hypothesis: Trade breakouts of 12-hour price channels with volume confirmation and MA trend filter.
# Long when price breaks above 12h high with volume surge and 12h EMA50 uptrend.
# Short when price breaks below 12h low with volume surge and 12h EMA50 downtrend.
# Designed for 4h timeframe to target 20-50 trades/year (80-200 total over 4 years).
# Uses 12h timeframe for signal generation to reduce frequency and avoid fee drag.
# Works in both bull and bear markets by following the 12h trend.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_breakout_volume_ma_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 12h data for price channels and trend
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # EMA50 for 12h trend
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Price channels: 20-period high/low
    high_ch = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    low_ch = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    
    # Align 12h data to 4h timeframe
    high_ch_aligned = align_htf_to_ltf(prices, df_12h, high_ch)
    low_ch_aligned = align_htf_to_ltf(prices, df_12h, low_ch)
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 50  # Ensure EMA50 is ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(high_ch_aligned[i]) or np.isnan(low_ch_aligned[i]) or 
            np.isnan(ema50_12h_aligned[i]) or np.isnan(vol_ma_20_aligned[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        # Volume surge condition
        vol_surge = volume[i] > 1.5 * vol_ma_20_aligned[i] if vol_ma_20_aligned[i] > 0 else False
        
        if position == 1:  # Long position
            # Exit: price breaks below 12h EMA50
            if close[i] < ema50_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above 12h EMA50
            if close[i] > ema50_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price breaks above 12h channel with volume surge and 12h EMA50 uptrend
            if (close[i] > high_ch_aligned[i] and vol_surge and 
                close[i] > ema50_12h_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below 12h channel with volume surge and 12h EMA50 downtrend
            elif (close[i] < low_ch_aligned[i] and vol_surge and 
                  close[i] < ema50_12h_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals