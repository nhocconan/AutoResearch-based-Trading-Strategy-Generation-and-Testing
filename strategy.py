#!/usr/bin/env python3
# 12h_1w_1d_price_channel_breakout_volume
# Hypothesis: Trade breakouts of weekly Donchian channels (period=20) with volume confirmation on daily timeframe.
# Uses weekly price channels to capture medium-term trends, with daily volume surge to confirm breakout strength.
# Long when price breaks above weekly upper channel with volume surge and weekly uptrend (weekly close > weekly SMA50).
# Short when price breaks below weekly lower channel with volume surge and weekly downtrend (weekly close < weekly SMA50).
# Designed for 12h timeframe to target 12-37 trades/year (50-150 total over 4 years).
# Weekly trend filter ensures alignment with higher timeframe momentum, working in both bull and bear markets.
# Uses discrete position sizes (0.0, ±0.25) to minimize fee churn.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1w_1d_price_channel_breakout_volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Weekly data for Donchian channels
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly Donchian channels (period=20)
    upper_1w = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    lower_1w = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Weekly SMA50 for trend filter
    sma50_1w = pd.Series(close_1w).rolling(window=50, min_periods=50).mean().values
    
    # Align weekly channels to 12h timeframe
    upper_aligned = align_htf_to_ltf(prices, df_1w, upper_1w)
    lower_aligned = align_htf_to_ltf(prices, df_1w, lower_1w)
    sma50_aligned = align_htf_to_ltf(prices, df_1w, sma50_1w)
    
    # Daily volume confirmation: volume > 1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 50  # Ensure SMA50 is ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or 
            np.isnan(sma50_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        # Volume surge condition
        vol_surge = volume[i] > 1.8 * vol_ma_20[i] if vol_ma_20[i] > 0 else False
        
        if position == 1:  # Long position
            # Exit: price breaks below weekly lower channel
            if close[i] < lower_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above weekly upper channel
            if close[i] > upper_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price breaks above weekly upper channel with volume surge and weekly uptrend
            if (close[i] > upper_aligned[i] and vol_surge and 
                close_1w[-1] > sma50_1w[-1] if len(close_1w) > 0 else False):  # Weekly close > SMA50
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below weekly lower channel with volume surge and weekly downtrend
            elif (close[i] < lower_aligned[i] and vol_surge and 
                  close_1w[-1] < sma50_1w[-1] if len(close_1w) > 0 else False):  # Weekly close < SMA50
                position = -1
                signals[i] = -0.25
    
    return signals