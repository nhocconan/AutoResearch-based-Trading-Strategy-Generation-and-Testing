#!/usr/bin/env python3
"""
Hypothesis: 1d timeframe with weekly ATR-based volatility filter + Donchian(20) breakout + volume confirmation.
Long when price breaks above 20-day Donchian high with weekly ATR ratio > 1.2 and volume > 1.5x 20-day average.
Short when price breaks below 20-day Donchian low with weekly ATR ratio > 1.2 and volume > 1.5x 20-day average.
Uses discrete position sizing 0.25 to limit fee drag. Target: 30-100 total trades over 4 years.
Weekly ATR filter ensures we only trade during sufficient volatility regimes, reducing whipsaw in sideways markets.
Donchian breakouts provide clear structural levels; volume confirms institutional participation.
Designed to work in bull markets (breakout continuation) and bear markets (volatility expansion after panic).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Donchian and volume
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Get 1w data for ATR filter
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1d Donchian channels (20-period)
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d volume 20-period average
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1w ATR (14-period) for volatility filter
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1w = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_ma_50_1w = pd.Series(atr_14_1w).rolling(window=50, min_periods=50).mean().values
    atr_ratio = atr_14_1w / atr_ma_50_1w  # Current ATR vs long-term average
    
    # Align all to 1d
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1w, atr_ratio)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # need enough for Donchian, volume MA, and ATR ratio
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(vol_ma_20_1d_aligned[i]) or np.isnan(volume_1d_aligned[i]) or 
            np.isnan(atr_ratio_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: weekly ATR ratio > 1.2 (sufficient volatility)
        volatility_filter = atr_ratio_aligned[i] > 1.2
        
        # Volume confirmation: current 1d volume > 1.5x 20-day average
        volume_confirmed = volume_1d_aligned[i] > 1.5 * vol_ma_20_1d_aligned[i]
        
        if position == 0:
            # Long: price breaks above Donchian high with volatility and volume
            if (close[i] > donchian_high_aligned[i] and 
                volatility_filter and 
                volume_confirmed):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low with volatility and volume
            elif (close[i] < donchian_low_aligned[i] and 
                  volatility_filter and 
                  volume_confirmed):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls back below Donchian low (mean reversion or failed breakout)
            if close[i] < donchian_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises back above Donchian high (mean reversion or failed breakdown)
            if close[i] > donchian_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_1wATR_VolFilter_Donchian20_Breakout_Volume"
timeframe = "1d"
leverage = 1.0