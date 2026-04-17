#!/usr/bin/env python3
"""
Hypothesis: 12h timeframe with daily Donchian(20) breakout + volume confirmation + 1d EMA200 trend filter.
Long when price breaks above 20-period Donchian high with volume > 1.3x 20-period average and close > EMA200.
Short when price breaks below 20-period Donchian low with volume > 1.3x 20-period average and close < EMA200.
Donchian channels capture volatility-based support/resistance; breakouts with volume and trend filter reduce false signals.
Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag. Uses discrete sizing 0.25.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Donchian channels, EMA, and volume
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 20-period Donchian channels
    high_ma_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).mean().values
    low_ma_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).mean().values
    # Donchian high = rolling max of high, Donchian low = rolling min of low
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d EMA200 for trend filter
    close_1d_series = pd.Series(close_1d)
    ema200_1d = close_1d_series.ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Calculate 1d volume 20-period average
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all to 12h
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 200  # need enough for EMA200
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema200_1d_aligned[i]) or np.isnan(vol_ma_20_1d_aligned[i]) or 
            np.isnan(volume_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 1.3x 20-period average
        volume_confirmed = volume_1d_aligned[i] > 1.3 * vol_ma_20_1d_aligned[i]
        
        if position == 0:
            # Long: price breaks above Donchian high with volume and bullish trend (close > EMA200)
            if (close[i] > donchian_high_aligned[i] and 
                volume_confirmed and 
                close[i] > ema200_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low with volume and bearish trend (close < EMA200)
            elif (close[i] < donchian_low_aligned[i] and 
                  volume_confirmed and 
                  close[i] < ema200_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls back below Donchian low or trend turns bearish
            if (close[i] < donchian_low_aligned[i] or 
                close[i] < ema200_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises back above Donchian high or trend turns bullish
            if (close[i] > donchian_high_aligned[i] or 
                close[i] > ema200_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1dDonchian20_Volume_EMA200"
timeframe = "12h"
leverage = 1.0