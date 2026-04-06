#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1w trend filter and volume confirmation.
# Long when price breaks above Donchian upper band during bullish weekly trend with volume > 1.5x average.
# Short when price breaks below Donchian lower band during bearish weekly trend with volume > 1.5x average.
# Weekly trend filter prevents counter-trend trades. Volume confirmation adds conviction.
# Target: 50-150 total trades over 4 years (12-37/year) to stay within optimal range.

name = "6h_donchian20_1w_vol_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 25:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian Channel (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    
    donchian_upper = high_series.rolling(window=20, min_periods=20).max()
    donchian_lower = low_series.rolling(window=20, min_periods=20).min()
    
    # Weekly trend filter: bullish/bearish week based on close vs open
    df_1w = get_htf_data(prices, '1w')
    weekly_open = df_1w['open'].values
    weekly_close = df_1w['close'].values
    weekly_bullish = weekly_close > weekly_open  # True for bullish week
    weekly_bearish = weekly_close < weekly_open   # True for bearish week
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bullish)
    weekly_bearish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bearish)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_series = pd.Series(volume)
    volume_ma = volume_series.rolling(window=20, min_periods=20).mean()
    volume_threshold = volume_ma * 1.5
    volume_confirmed = volume > volume_threshold
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if weekly trend data not available
        if np.isnan(weekly_bullish_aligned[i]) or np.isnan(weekly_bearish_aligned[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits
        if position == 1:  # long position
            # Exit: price falls below Donchian lower band or weekly turn bearish
            if (close[i] < donchian_lower[i] or 
                weekly_bearish_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price rises above Donchian upper band or weekly turn bullish
            if (close[i] > donchian_upper[i] or 
                weekly_bullish_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with weekly trend filter and volume confirmation
            # Long: price breaks above Donchian upper band during bullish week with volume confirmation
            if (close[i] > donchian_upper[i] and 
                weekly_bullish_aligned[i] and 
                volume_confirmed[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower band during bearish week with volume confirmation
            elif (close[i] < donchian_lower[i] and 
                  weekly_bearish_aligned[i] and 
                  volume_confirmed[i]):
                signals[i] = -0.25
                position = -1
    
    return signals