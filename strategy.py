#!/usr/bin/env python3
# 1d_1w_price_channel_breakout
# Hypothesis: Breakouts above weekly Donchian high or below weekly Donchian low with
# volume confirmation and daily KAMA trend filter. Works in bull markets (breakouts)
# and bear markets (breakdowns) by capturing volatility expansion. Low trade frequency
# via weekly channel and volume filter minimizes fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_price_channel_breakout"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Weekly Donchian channels for breakout levels
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate 20-period Donchian channels on weekly data
    high_max_20 = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Align weekly Donchian levels to daily timeframe (wait for weekly close)
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, high_max_20)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, low_min_20)
    
    # Daily KAMA for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate Efficiency Ratio for KAMA
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.abs(np.diff(close_1d))
    er = np.divide(change, volatility, out=np.zeros_like(change), where=volatility!=0)
    er = np.clip(er, 0, 1)
    
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    
    # Calculate KAMA
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # Align daily KAMA to daily timeframe (no shift needed as it's same TF)
    kama_aligned = kama  # already on daily
    
    # Volume confirmation: current volume > 1.5x 20-day average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 40  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(kama_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        # Volume surge condition
        vol_surge = volume[i] > 1.5 * vol_ma_20[i] if vol_ma_20[i] > 0 else False
        
        if position == 1:  # Long position
            # Exit: price below weekly Donchian low OR KAMA turns down
            if close[i] < donchian_low_aligned[i] or close[i] < kama_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price above weekly Donchian high OR KAMA turns up
            if close[i] > donchian_high_aligned[i] or close[i] > kama_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price breaks above weekly Donchian high with volume surge and above KAMA
            if close[i] > donchian_high_aligned[i] and vol_surge and close[i] > kama_aligned[i]:
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below weekly Donchian low with volume surge and below KAMA
            elif close[i] < donchian_low_aligned[i] and vol_surge and close[i] < kama_aligned[i]:
                position = -1
                signals[i] = -0.25
    
    return signals