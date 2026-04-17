#!/usr/bin/env python3
"""
Hypothesis: 4h timeframe with 1d Donchian(20) breakout + volume confirmation + 1d EMA50/EMA200 trend filter.
Long when price breaks above 1d Donchian upper channel with volume > 1.3x 20-period average and EMA50 > EMA200.
Short when price breaks below 1d Donchian lower channel with volume > 1.3x 20-period average and EMA50 < EMA200.
Exit on opposite Donchian channel touch or trend filter reversal.
Uses discrete sizing 0.25 to limit fee drag. Target: 50-150 total trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Donchian channels and EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d Donchian channels (20-period)
    high_ma_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_ma_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donchian_upper = high_ma_20
    donchian_lower = low_ma_20
    
    # Calculate 1d EMA50 and EMA200 for trend filter
    close_1d_series = pd.Series(close_1d)
    ema50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema200_1d = close_1d_series.ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Calculate 1d volume 20-period average
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all to 4h
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 200  # need enough for EMA200 and Donchian
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(ema200_1d_aligned[i]) or 
            np.isnan(vol_ma_20_1d_aligned[i]) or np.isnan(volume_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 1.3x 20-period average
        volume_confirmed = volume_1d_aligned[i] > 1.3 * vol_ma_20_1d_aligned[i]
        
        if position == 0:
            # Long: price breaks above 1d Donchian upper with volume and bullish trend
            if (close[i] > donchian_upper_aligned[i] and 
                volume_confirmed and 
                ema50_1d_aligned[i] > ema200_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 1d Donchian lower with volume and bearish trend
            elif (close[i] < donchian_lower_aligned[i] and 
                  volume_confirmed and 
                  ema50_1d_aligned[i] < ema200_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls back to 1d Donchian lower or trend turns bearish
            if (close[i] < donchian_lower_aligned[i] or 
                ema50_1d_aligned[i] < ema200_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises back to 1d Donchian upper or trend turns bullish
            if (close[i] > donchian_upper_aligned[i] or 
                ema50_1d_aligned[i] > ema200_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_1dDonchian20_Volume_EMATrend"
timeframe = "4h"
leverage = 1.0