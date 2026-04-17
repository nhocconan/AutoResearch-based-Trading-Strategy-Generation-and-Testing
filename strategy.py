#!/usr/bin/env python3
"""
Hypothesis: 1h timeframe with 4h Donchian breakout (20-period) + 1d EMA trend filter + volume confirmation.
Long when price breaks above 4h Donchian upper band with volume > 1.3x 20-period 1h average and 1d EMA50 > EMA200.
Short when price breaks below 4h Donchian lower band with volume > 1.3x 20-period 1h average and 1d EMA50 < EMA200.
4h Donchian provides structure, 1d EMA filters trend, volume confirms breakout strength.
Session filter (08-20 UTC) reduces noise. Target: 60-150 total trades over 4 years (15-37/year) to avoid fee drag.
Uses discrete sizing 0.20. Works in both bull (trend following) and bear (mean reversion via trend filter).
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
    
    # Get 4h data for Donchian channels
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate 4h Donchian channels (20-period)
    high_4h_series = pd.Series(high_4h)
    low_4h_series = pd.Series(low_4h)
    donchian_upper_4h = high_4h_series.rolling(window=20, min_periods=20).max().values
    donchian_lower_4h = low_4h_series.rolling(window=20, min_periods=20).min().values
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 and EMA200
    close_1d_series = pd.Series(close_1d)
    ema50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema200_1d = close_1d_series.ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Get 1h volume 20-period average
    vol_ma_20_1h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align all to 1h
    donchian_upper_4h_aligned = align_htf_to_ltf(prices, df_4h, donchian_upper_4h)
    donchian_lower_4h_aligned = align_htf_to_ltf(prices, df_4h, donchian_lower_4h)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    start_idx = 200  # need enough for EMA200 and Donchian
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Skip if any required data is not available
        if (np.isnan(donchian_upper_4h_aligned[i]) or np.isnan(donchian_lower_4h_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(ema200_1d_aligned[i]) or 
            np.isnan(vol_ma_20_1h[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1h volume > 1.3x 20-period average
        volume_confirmed = volume[i] > 1.3 * vol_ma_20_1h[i]
        
        if position == 0:
            # Long: price breaks above 4h Donchian upper with volume and bullish trend (EMA50 > EMA200)
            if (close[i] > donchian_upper_4h_aligned[i] and 
                volume_confirmed and 
                ema50_1d_aligned[i] > ema200_1d_aligned[i]):
                signals[i] = 0.20
                position = 1
            # Short: price breaks below 4h Donchian lower with volume and bearish trend (EMA50 < EMA200)
            elif (close[i] < donchian_lower_4h_aligned[i] and 
                  volume_confirmed and 
                  ema50_1d_aligned[i] < ema200_1d_aligned[i]):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: price falls back below 4h Donchian middle or trend turns bearish
            donchian_middle_4h = (donchian_upper_4h_aligned[i] + donchian_lower_4h_aligned[i]) / 2
            if (close[i] < donchian_middle_4h or 
                ema50_1d_aligned[i] < ema200_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: price rises back above 4h Donchian middle or trend turns bullish
            donchian_middle_4h = (donchian_upper_4h_aligned[i] + donchian_lower_4h_aligned[i]) / 2
            if (close[i] > donchian_middle_4h or 
                ema50_1d_aligned[i] > ema200_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_4hDonchian20_Volume_1dEMA"
timeframe = "1h"
leverage = 1.0