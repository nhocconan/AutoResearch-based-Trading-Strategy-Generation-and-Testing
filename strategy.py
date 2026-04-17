#!/usr/bin/env python3
"""
Hypothesis: 12h timeframe with weekly Donchian breakout + volume confirmation + 1d EMA trend filter.
Long when price breaks above weekly Donchian high (20) with volume > 1.5x 20-period average and 1d EMA50 > EMA200.
Short when price breaks below weekly Donchian low (20) with volume > 1.5x 20-period average and 1d EMA50 < EMA200.
Weekly Donchian channels capture major institutional levels; breakouts with volume and trend filter reduce false signals.
Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag. Uses discrete sizing 0.25.
Works in both bull and bear markets by requiring volume confirmation and trend alignment.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Donchian channels
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly Donchian channels (20-period)
    high_1w_series = pd.Series(high_1w)
    low_1w_series = pd.Series(low_1w)
    donchian_high_20 = high_1w_series.rolling(window=20, min_periods=20).max().values
    donchian_low_20 = low_1w_series.rolling(window=20, min_periods=20).min().values
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d EMA50 and EMA200
    close_1d_series = pd.Series(close_1d)
    ema50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema200_1d = close_1d_series.ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Calculate 1d volume 20-period average
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all to 12h
    donchian_high_20_aligned = align_htf_to_ltf(prices, df_1w, donchian_high_20)
    donchian_low_20_aligned = align_htf_to_ltf(prices, df_1w, donchian_low_20)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 200  # need enough for EMA200 and Donchian
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_high_20_aligned[i]) or np.isnan(donchian_low_20_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(ema200_1d_aligned[i]) or 
            np.isnan(vol_ma_20_1d_aligned[i]) or np.isnan(volume_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 1.5x 20-period average
        volume_confirmed = volume_1d_aligned[i] > 1.5 * vol_ma_20_1d_aligned[i]
        
        if position == 0:
            # Long: price breaks above weekly Donchian high with volume and bullish trend (EMA50 > EMA200)
            if (close[i] > donchian_high_20_aligned[i] and 
                volume_confirmed and 
                ema50_1d_aligned[i] > ema200_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly Donchian low with volume and bearish trend (EMA50 < EMA200)
            elif (close[i] < donchian_low_20_aligned[i] and 
                  volume_confirmed and 
                  ema50_1d_aligned[i] < ema200_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls back below weekly Donchian low or trend turns bearish
            if (close[i] < donchian_low_20_aligned[i] or 
                ema50_1d_aligned[i] < ema200_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises back above weekly Donchian high or trend turns bullish
            if (close[i] > donchian_high_20_aligned[i] or 
                ema50_1d_aligned[i] > ema200_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1wDonchian20_Volume_1dEMA"
timeframe = "12h"
leverage = 1.0