#!/usr/bin/env python3
"""
Hypothesis: 12h timeframe with 1d Donchian(20) breakout + volume confirmation + EMA50 trend filter.
Long when price breaks above 1d Donchian high with volume > 1.5x 20-period average and close > EMA50.
Short when price breaks below 1d Donchian low with volume > 1.5x 20-period average and close < EMA50.
Exit when price returns to Donchian midpoint or volume drops below average.
Uses discrete sizing 0.25 to minimize fee churn. Target: 50-150 total trades over 4 years.
Works in bull markets via breakout continuation and in bear markets via mean reversion after volatility spikes.
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
    
    # Get daily data for Donchian channels, volume, and EMA50
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate daily Donchian channels (20-period)
    high_series = pd.Series(high_1d)
    low_series = pd.Series(low_1d)
    upper_20 = high_series.rolling(window=20, min_periods=20).max().values
    lower_20 = low_series.rolling(window=20, min_periods=20).min().values
    
    # Calculate daily EMA50
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Get 1d volume 20-period average
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all to 12h
    upper_20_aligned = align_htf_to_ltf(prices, df_1d, upper_20)
    lower_20_aligned = align_htf_to_ltf(prices, df_1d, lower_20)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 70  # need enough for EMA50 and Donchian
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper_20_aligned[i]) or np.isnan(lower_20_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma_20_1d_aligned[i]) or 
            np.isnan(volume_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 1.5x 20-period average
        volume_confirmed = volume_1d_aligned[i] > 1.5 * vol_ma_20_1d_aligned[i]
        
        if position == 0:
            # Long: price breaks above daily Donchian high with volume and above EMA50
            if (close[i] > upper_20_aligned[i] and 
                volume_confirmed and 
                close[i] > ema_50_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below daily Donchian low with volume and below EMA50
            elif (close[i] < lower_20_aligned[i] and 
                  volume_confirmed and 
                  close[i] < ema_50_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls back below daily Donchian midpoint or volume drops
            midpoint_20 = (upper_20 + lower_20) / 2
            midpoint_20_aligned = align_htf_to_ltf(prices, df_1d, midpoint_20)
            if (close[i] < midpoint_20_aligned[i] or 
                volume_1d_aligned[i] < vol_ma_20_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises back above daily Donchian midpoint or volume drops
            midpoint_20 = (upper_20 + lower_20) / 2
            midpoint_20_aligned = align_htf_to_ltf(prices, df_1d, midpoint_20)
            if (close[i] > midpoint_20_aligned[i] or 
                volume_1d_aligned[i] < vol_ma_20_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1dDonchian20_Volume_EMA50"
timeframe = "12h"
leverage = 1.0