#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-day Donchian channel breakout with weekly trend filter and volume confirmation.
# Long when: Price breaks above 20-day high, weekly close > weekly open (bullish week), volume > 1.5x 20-day average
# Short when: Price breaks below 20-day low, weekly close < weekly open (bearish week), volume > 1.5x 20-day average
# Exit when: Price crosses back through the 20-day midpoint
# Donchian provides clear breakout levels, weekly trend filters counter-trend noise, volume confirms conviction.
# Target: 15-25 trades/year per symbol. Works in bull (buy strength) and bear (sell weakness).
name = "1d_Donchian20_WeeklyTrend_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    weekly_close = df_1w['close'].values
    weekly_open = df_1w['open'].values
    
    # Calculate 20-day Donchian channels
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (high_20 + low_20) / 2.0
    
    # Weekly trend: bullish if weekly close > open, bearish if close < open
    weekly_bullish = weekly_close > weekly_open
    weekly_bearish = weekly_close < weekly_open
    
    # Align weekly data to daily timeframe
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bullish.astype(float))
    weekly_bearish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bearish.astype(float))
    
    # 20-day volume average for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Wait for Donchian calculation
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(weekly_bullish_aligned[i]) or np.isnan(weekly_bearish_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        high_20_val = high_20[i]
        low_20_val = low_20[i]
        mid = donchian_mid[i]
        weekly_bull = weekly_bullish_aligned[i]
        weekly_bear = weekly_bearish_aligned[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        if position == 0:
            # Long entry: Price breaks above 20-day high, weekly bullish, volume spike
            if (price > high_20_val and close[i-1] <= high_20_val and 
                weekly_bull > 0.5 and vol > 1.5 * vol_ma):
                signals[i] = 0.25
                position = 1
            # Short entry: Price breaks below 20-day low, weekly bearish, volume spike
            elif (price < low_20_val and close[i-1] >= low_20_val and 
                  weekly_bear > 0.5 and vol > 1.5 * vol_ma):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price crosses back below 20-day midpoint
            if price < mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price crosses back above 20-day midpoint
            if price > mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals