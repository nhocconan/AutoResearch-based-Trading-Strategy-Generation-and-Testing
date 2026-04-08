#!/usr/bin/env python3
# 12h_1d_1w_price_channel_breakout_volume_v1
# Hypothesis: Trade breakouts of daily price channels with volume confirmation on 12h timeframe.
# Uses daily Donchian channels for breakout signals, volume surge to confirm breakout strength,
# and weekly trend filter (price > weekly EMA50 for long, < weekly EMA50 for short).
# Designed for 12h timeframe to target 12-37 trades/year (50-150 total over 4 years).
# Weekly trend filter ensures alignment with higher timeframe momentum, working in both bull and bear markets.
# Low trade frequency minimizes fee drag while capturing significant moves.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_1w_price_channel_breakout_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily data for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Daily Donchian channels (20-period)
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # EMA50 for weekly trend
    ema50 = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align daily Donchian channels to 12h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # Align weekly EMA50 to 12h timeframe
    ema50_aligned = align_htf_to_ltf(prices, df_1w, ema50)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 50  # Ensure EMA50 is ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema50_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        # Volume surge condition
        vol_surge = volume[i] > 1.5 * vol_ma_20[i] if vol_ma_20[i] > 0 else False
        
        if position == 1:  # Long position
            # Exit: price breaks below daily Donchian low
            if close[i] < donchian_low_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above daily Donchian high
            if close[i] > donchian_high_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price breaks above daily Donchian high with volume surge and weekly uptrend
            if (close[i] > donchian_high_aligned[i] and vol_surge and 
                close[i] > ema50_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below daily Donchian low with volume surge and weekly downtrend
            elif (close[i] < donchian_low_aligned[i] and vol_surge and 
                  close[i] < ema50_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals