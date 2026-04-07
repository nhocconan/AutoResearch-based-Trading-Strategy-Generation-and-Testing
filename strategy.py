#!/usr/bin/env python3
"""
4h_donchian_20_1d_trend_volume_v3
Hypothesis: On 4-hour timeframe, use Donchian channel breakout with 1-day trend filter and volume confirmation.
Enter long when price breaks above 20-period high and 1d close > 1d open (bullish day) with volume > 1.5x average.
Enter short when price breaks below 20-period low and 1d close < 1d open (bearish day) with volume > 1.5x average.
Exit on opposite Donchian break or volume failure.
Designed for 20-50 trades/year to minimize fee drag while capturing strong directional moves.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_20_1d_trend_volume_v3"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Daily trend: bullish if close > open, bearish if close < open
    d_close = df_1d['close'].values
    d_open = df_1d['open'].values
    daily_bullish = d_close > d_open
    daily_bearish = d_close < d_open
    
    # Align daily trend to 4h
    bullish_aligned = align_htf_to_ltf(prices, df_1d, daily_bullish)
    bearish_aligned = align_htf_to_ltf(prices, df_1d, daily_bearish)
    
    # Donchian channel (20-period) on 4h
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume average (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if data not ready
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or \
           np.isnan(bullish_aligned[i]) or np.isnan(bearish_aligned[i]) or \
           np.isnan(vol_avg[i]):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_avg[i]
        
        if position == 1:  # Long position
            # Exit: price breaks below Donchian low OR no longer bullish day
            if close[i] < donchian_low[i] or not bullish_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.30
                
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian high OR no longer bearish day
            if close[i] > donchian_high[i] or not bearish_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.30
        else:  # Flat, look for entry
            # Long entry: price breaks above Donchian high + bullish day + volume
            long_entry = (close[i] > donchian_high[i]) and bullish_aligned[i] and vol_confirm
            # Short entry: price breaks below Donchian low + bearish day + volume
            short_entry = (close[i] < donchian_low[i]) and bearish_aligned[i] and vol_confirm
            
            if long_entry:
                position = 1
                signals[i] = 0.30
            elif short_entry:
                position = -1
                signals[i] = -0.30
    
    return signals