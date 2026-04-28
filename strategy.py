#!/usr/bin/env python3
# Hypothesis: 1d Donchian(20) breakout with 1w SMA50 trend filter and volume confirmation.
# Donchian channels identify breakouts of recent price extremes, capturing momentum.
# Weekly SMA50 filters for higher timeframe trend alignment, avoiding counter-trend trades.
# Volume spike (2x 20-period average) confirms breakout validity, reducing false signals.
# Works in bull markets (catching uptrends via upper band breakouts) and bear markets (catching downtrends via lower band breakdowns).
# Targets 30-100 total trades over 4 years (7-25/year) with discrete position sizing to minimize fee drag.

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
    
    # Get 1w data for SMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w SMA(50)
    close_1w = df_1w['close'].values
    sma_50_1w = pd.Series(close_1w).rolling(window=50, min_periods=50).mean().values
    sma_50_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_50_1w)
    
    # Calculate Donchian channels from daily OHLC (20-period)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Upper band: 20-day high, Lower band: 20-day low
    upper_band = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lower_band = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian bands to daily timeframe (wait for prior day's close)
    upper_band_aligned = align_htf_to_ltf(prices, df_1d, upper_band)
    lower_band_aligned = align_htf_to_ltf(prices, df_1d, lower_band)
    
    # Volume filter: volume > 2x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Wait for SMA and Donchian
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(sma_50_1w_aligned[i]) or np.isnan(upper_band_aligned[i]) or 
            np.isnan(lower_band_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 1w SMA(50)
        uptrend = close[i] > sma_50_1w_aligned[i]
        downtrend = close[i] < sma_50_1w_aligned[i]
        
        # Donchian breakout conditions
        breakout_upper = high[i] > upper_band_aligned[i-1]  # Break above upper band
        breakdown_lower = low[i] < lower_band_aligned[i-1]  # Break below lower band
        
        # Entry conditions with volume spike confirmation
        long_entry = uptrend and breakout_upper and volume_spike[i]
        short_entry = downtrend and breakdown_lower and volume_spike[i]
        
        # Exit conditions: trend reversal or opposite Donchian break
        long_exit = (not uptrend) or breakdown_lower
        short_exit = (not downtrend) or breakout_upper
        
        # Handle entries and exits
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_DonchianBreakout_1wSMA50_Trend_VolumeSpike"
timeframe = "1d"
leverage = 1.0