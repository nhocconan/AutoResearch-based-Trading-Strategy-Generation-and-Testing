#!/usr/bin/env python3
"""
Hypothesis: 12h timeframe with 1d Donchian channel breakout + volume confirmation + 1w EMA trend filter.
Long when price breaks above 1d Donchian upper (20) with volume > 1.3x 20-period average and 1w EMA50 > EMA200.
Short when price breaks below 1d Donchian lower (20) with volume > 1.3x 20-period average and 1w EMA50 < EMA200.
Donchian channels capture volatility-based breakouts; volume confirmation reduces false signals; 1w EMA filter ensures alignment with major trend.
Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag. Uses discrete sizing 0.25.
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
    
    # Get 1d data for Donchian channels and volume
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d Donchian channels (20-period)
    high_ma_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).mean().values
    low_ma_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).mean().values
    # Upper band = highest high over 20 periods
    donchian_upper = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    # Lower band = lowest low over 20 periods
    donchian_lower = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d volume 20-period average
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Get 1w data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA50 and EMA200
    close_1w_series = pd.Series(close_1w)
    ema50_1w = close_1w_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema200_1w = close_1w_series.ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align all to 12h
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 200  # need enough for EMA200 and Donchian
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(vol_ma_20_aligned[i]) or np.isnan(volume_1d_aligned[i]) or 
            np.isnan(ema50_1w_aligned[i]) or np.isnan(ema200_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 1.3x 20-period average
        volume_confirmed = volume_1d_aligned[i] > 1.3 * vol_ma_20_aligned[i]
        
        if position == 0:
            # Long: price breaks above 1d Donchian upper with volume and bullish trend (EMA50 > EMA200)
            if (close[i] > donchian_upper_aligned[i] and 
                volume_confirmed and 
                ema50_1w_aligned[i] > ema200_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 1d Donchian lower with volume and bearish trend (EMA50 < EMA200)
            elif (close[i] < donchian_lower_aligned[i] and 
                  volume_confirmed and 
                  ema50_1w_aligned[i] < ema200_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls back below 1d Donchian lower or trend turns bearish
            if (close[i] < donchian_lower_aligned[i] or 
                ema50_1w_aligned[i] < ema200_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises back above 1d Donchian upper or trend turns bullish
            if (close[i] > donchian_upper_aligned[i] or 
                ema50_1w_aligned[i] > ema200_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1dDonchian20_Volume_1wEMA"
timeframe = "12h"
leverage = 1.0