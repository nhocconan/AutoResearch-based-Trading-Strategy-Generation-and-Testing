#!/usr/bin/env python3
# 12h_1d_1w_donchian_breakout_volume_v1
# Hypothesis: Trade Donchian channel breakouts on 12h timeframe with daily volume confirmation and weekly trend filter.
# Long when price breaks above 12h Donchian(20) upper band with volume surge and weekly uptrend (price > weekly EMA50).
# Short when price breaks below 12h Donchian(20) lower band with volume surge and weekly downtrend (price < weekly EMA50).
# Designed for 12h timeframe to target 12-37 trades/year (50-150 total over 4 years).
# Weekly trend filter ensures alignment with higher timeframe momentum, working in both bull and bear markets.
# Volume surge filters out false breakouts, reducing whipsaws in ranging markets.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_1w_donchian_breakout_volume_v1"
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
    
    # Daily data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    
    # Weekly data for trend filter and Donchian channels
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate ATR(10) for weekly data (used in Donchian bands)
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    # EMA50 for weekly trend filter
    ema50 = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Donchian channels (20-period) on weekly data
    donch_high = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Align weekly data to 12h timeframe
    donch_high_aligned = align_htf_to_ltf(prices, df_1w, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_1w, donch_low)
    ema50_aligned = align_htf_to_ltf(prices, df_1w, ema50)
    
    # Volume confirmation: volume > 1.5x 20-day average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 50  # Ensure EMA50 is ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or 
            np.isnan(ema50_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        # Volume surge condition
        vol_surge = volume[i] > 1.5 * vol_ma_20[i] if vol_ma_20[i] > 0 else False
        
        if position == 1:  # Long position
            # Exit: price breaks below Donchian lower band
            if close[i] < donch_low_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian upper band
            if close[i] > donch_high_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price breaks above Donchian upper band with volume surge and weekly uptrend
            if (close[i] > donch_high_aligned[i] and vol_surge and 
                close[i] > ema50_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below Donchian lower band with volume surge and weekly downtrend
            elif (close[i] < donch_low_aligned[i] and vol_surge and 
                  close[i] < ema50_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals