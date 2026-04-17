#!/usr/bin/env python3
"""
Hypothesis: 12h timeframe with daily Donchian channel breakout + volume confirmation + weekly EMA trend filter.
Long when price breaks above daily Donchian(20) upper band with volume > 1.3x 20-period average and weekly EMA50 > EMA200.
Short when price breaks below daily Donchian(20) lower band with volume > 1.3x 20-period average and weekly EMA50 < EMA200.
Daily Donchian captures intermediate-term structure; breakouts with volume and weekly trend filter reduce false signals.
Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag. Uses discrete sizing 0.25.
Works in both bull and bear markets: Donchian breakouts catch strong moves, volume confirms participation, weekly EMA ensures alignment with higher timeframe trend.
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
    
    # Get daily data for Donchian channels and volume
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate daily Donchian(20) channels
    # Upper band = 20-period high, Lower band = 20-period low
    high_series = pd.Series(high_1d)
    low_series = pd.Series(low_1d)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Calculate daily volume 20-period average
    volume_series = pd.Series(volume_1d)
    vol_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    
    # Get weekly data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA50 and EMA200
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
    
    start_idx = 200  # need enough for weekly EMA200
    
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
            # Long: price breaks above daily Donchian upper with volume and bullish weekly trend (EMA50 > EMA200)
            if (close[i] > donchian_upper_aligned[i] and 
                volume_confirmed and 
                ema50_1w_aligned[i] > ema200_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below daily Donchian lower with volume and bearish weekly trend (EMA50 < EMA200)
            elif (close[i] < donchian_lower_aligned[i] and 
                  volume_confirmed and 
                  ema50_1w_aligned[i] < ema200_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls back below daily Donchian middle or weekly trend turns bearish
            donchian_middle = (donchian_upper + donchian_lower) / 2
            donchian_middle_aligned = align_htf_to_ltf(prices, df_1d, donchian_middle)
            if (close[i] < donchian_middle_aligned[i] or 
                ema50_1w_aligned[i] < ema200_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises back above daily Donchian middle or weekly trend turns bullish
            donchian_middle = (donchian_upper + donchian_lower) / 2
            donchian_middle_aligned = align_htf_to_ltf(prices, df_1d, donchian_middle)
            if (close[i] > donchian_middle_aligned[i] or 
                ema50_1w_aligned[i] > ema200_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1dDonchian20_Volume_1wEMA"
timeframe = "12h"
leverage = 1.0