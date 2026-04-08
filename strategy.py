#!/usr/bin/env python3
# 12h_donchian_breakout_daily_trend_volume_v2
# Hypothesis: Uses 12h Donchian(20) breakout with daily trend filter (price > daily EMA200) and volume confirmation.
# Enters long when price breaks above Donchian upper band and close > daily EMA200 and volume > 1.5x average volume.
# Enters short when price breaks below Donchian lower band and close < daily EMA200 and volume > 1.5x average volume.
# Exits when price crosses back across Donchian middle band (20-day SMA) or volume condition fails.
# Designed for 12-37 trades/year on 12h to avoid fee drag. Works in bull/bear via trend-following with strong filters.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_donchian_breakout_daily_trend_volume_v2"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily data for EMA200 filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate Donchian channels (20-period) on 12h data
    highest_high = np.maximum.accumulate(high)
    lowest_low = np.minimum.accumulate(low)
    
    # For Donchian breakout, we need the highest high and lowest low over the past 20 periods
    # We'll calculate rolling max/min using a deque-like approach for efficiency
    donchian_period = 20
    upperband = np.full(n, np.nan)
    lowerband = np.full(n, np.nan)
    middleband = np.full(n, np.nan)
    
    # Calculate rolling max/min using pandas for simplicity and correctness
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    close_series = pd.Series(close)
    
    upperband = high_series.rolling(window=donchian_period, min_periods=donchian_period).max().values
    lowerband = low_series.rolling(window=donchian_period, min_periods=donchian_period).min().values
    middleband = close_series.rolling(window=donchian_period, min_periods=donchian_period).mean().values
    
    # Daily EMA200
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Average volume (20-period) for volume confirmation
    volume_series = pd.Series(volume)
    avg_volume = volume_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = max(200, 20)  # Ensure EMA200 and Donchian are ready
    
    for i in range(start_idx, n):
        if np.isnan(upperband[i]) or np.isnan(lowerband[i]) or np.isnan(middleband[i]) or np.isnan(ema200_1d_aligned[i]) or np.isnan(avg_volume[i]):
            if position != 0:
                pass
            else:
                signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirm = volume[i] > 1.5 * avg_volume[i]
        
        if position == 1:  # Long position
            # Exit: price crosses below middleband or volume confirmation fails
            if close[i] < middleband[i] or not volume_confirm:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above middleband or volume confirmation fails
            if close[i] > middleband[i] or not volume_confirm:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price breaks above upperband and close > daily EMA200 and volume confirmation
            if close[i] > upperband[i] and close[i] > ema200_1d_aligned[i] and volume_confirm:
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below lowerband and close < daily EMA200 and volume confirmation
            elif close[i] < lowerband[i] and close[i] < ema200_1d_aligned[i] and volume_confirm:
                position = -1
                signals[i] = -0.25
    
    return signals