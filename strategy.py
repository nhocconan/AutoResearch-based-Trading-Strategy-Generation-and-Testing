#!/usr/bin/env python3
# 6h_MultiTF_DonchianBreakout_WeeklyTrend_Volume
# Hypothesis: Uses 1-week Donchian channels for trend direction, 1-day Donchian breakouts for entry timing, and volume confirmation on 6h timeframe.
# Weekly trend filter ensures alignment with major market direction, reducing false breakouts in chop.
# Daily Donchian breakouts capture medium-term momentum, while volume spikes confirm institutional participation.
# Designed for 6h timeframe to balance trade frequency (target: 20-50/year) and signal quality.
# Works in bull markets via long breakouts in uptrend, and in bear markets via short breakdowns in downtrend.

name = "6h_MultiTF_DonchianBreakout_WeeklyTrend_Volume"
timeframe = "6h"
leverage = 1.0

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
    
    # Get 1w data for trend filter (Donchian channel)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate 20-period Donchian channel on weekly data
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Upper band: 20-period high
    donchian_high_20 = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    # Lower band: 20-period low  
    donchian_low_20 = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Determine trend: price above upper band = uptrend, below lower band = downtrend
    weekly_uptrend = high_1w > donchian_high_20
    weekly_downtrend = low_1w < donchian_low_20
    
    # Align weekly trend to 6h timeframe
    weekly_uptrend_6h = align_htf_to_ltf(prices, df_1w, weekly_uptrend.astype(float))
    weekly_downtrend_6h = align_htf_to_ltf(prices, df_1w, weekly_downtrend.astype(float))
    
    # Get 1d data for entry signal (Donchian breakout)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 20-period Donchian channel on daily data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    donchian_high_20_1d = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low_20_1d = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align daily Donchian levels to 6h timeframe
    donchian_high_20_1d_6h = align_htf_to_ltf(prices, df_1d, donchian_high_20_1d)
    donchian_low_20_1d_6h = align_htf_to_ltf(prices, df_1d, donchian_low_20_1d)
    
    # Calculate volume spike on 6h timeframe (30-period average)
    vol_ma_30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_spike = volume > (2.0 * vol_ma_30)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any critical value is NaN
        if (np.isnan(weekly_uptrend_6h[i]) or np.isnan(weekly_downtrend_6h[i]) or
            np.isnan(donchian_high_20_1d_6h[i]) or np.isnan(donchian_low_20_1d_6h[i]) or
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above daily Donchian high + weekly uptrend + volume spike
            if (close[i] > donchian_high_20_1d_6h[i] and 
                weekly_uptrend_6h[i] > 0.5 and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below daily Donchian low + weekly downtrend + volume spike
            elif (close[i] < donchian_low_20_1d_6h[i] and 
                  weekly_downtrend_6h[i] > 0.5 and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Price closes below daily Donchian low or weekly trend turns down
            if (close[i] < donchian_low_20_1d_6h[i] or 
                weekly_downtrend_6h[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Price closes above daily Donchian high or weekly trend turns up
            if (close[i] > donchian_high_20_1d_6h[i] or 
                weekly_uptrend_6h[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals