#!/usr/bin/env python3
"""
1d_donchian_20_1w_trend_volume_v1
Hypothesis: On daily timeframe, use Donchian channel breakout with weekly trend filter and volume confirmation.
Enter long when price breaks above 20-day high with weekly close > weekly open and volume > 1.5x average.
Enter short when price breaks below 20-day low with weekly close < weekly open and volume > 1.5x average.
Exit when price touches opposite Donchian band.
Designed for low frequency (7-25 trades/year) to minimize fee drag while capturing strong trends.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_donchian_20_1w_trend_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Weekly trend: close > open = uptrend, close < open = downtrend
    weekly_close = df_1w['close'].values
    weekly_open = df_1w['open'].values
    weekly_uptrend = weekly_close > weekly_open
    weekly_downtrend = weekly_close < weekly_open
    
    # Align weekly trend to daily timeframe
    weekly_uptrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_uptrend)
    weekly_downtrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_downtrend)
    
    # Calculate 20-day Donchian channels
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Calculate 20-period average volume for confirmation
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if weekly trend or Donchian data not available
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or \
           np.isnan(weekly_uptrend_aligned[i]) or np.isnan(weekly_downtrend_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume[i] > 1.5 * vol_avg[i] if not np.isnan(vol_avg[i]) else False
        
        if position == 1:  # Long position
            # Exit when price touches or goes below Donchian low
            if close[i] <= donchian_low[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit when price touches or goes above Donchian high
            if close[i] >= donchian_high[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price breaks above Donchian high with weekly uptrend and volume confirmation
            long_entry = (close[i] > donchian_high[i]) and weekly_uptrend_aligned[i] and vol_confirm
            # Short entry: price breaks below Donchian low with weekly downtrend and volume confirmation
            short_entry = (close[i] < donchian_low[i]) and weekly_downtrend_aligned[i] and vol_confirm
            
            if long_entry:
                position = 1
                signals[i] = 0.25
            elif short_entry:
                position = -1
                signals[i] = -0.25
    
    return signals