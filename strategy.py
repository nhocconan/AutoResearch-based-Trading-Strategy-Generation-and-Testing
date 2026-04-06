#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with volume confirmation and 1d trend filter.
# Long when price breaks above Donchian high (20) with above-average volume and bullish daily trend.
# Short when price breaks below Donchian low (20) with above-average volume and bearish daily trend.
# Uses daily trend filter to avoid counter-trend trades. Focus on breakouts with volume to avoid false signals.
# Target: 75-200 total trades over 4 years (19-50/year) to stay within optimal range.

name = "4h_donchian20_1d_vol_trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:  # Need at least 20 for Donchian + buffer
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian Channels (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max()
    donchian_low = low_series.rolling(window=20, min_periods=20).min()
    
    # Volume average (20-period)
    volume_series = pd.Series(volume)
    volume_ma = volume_series.rolling(window=20, min_periods=20).mean()
    
    # Daily trend filter: bullish/bearish day based on close vs open
    df_1d = get_htf_data(prices, '1d')
    daily_open = df_1d['open'].values
    daily_close = df_1d['close'].values
    daily_bullish = daily_close > daily_open  # True for bullish day
    daily_bearish = daily_close < daily_open   # True for bearish day
    daily_bullish_aligned = align_htf_to_ltf(prices, df_1d, daily_bullish)
    daily_bearish_aligned = align_htf_to_ltf(prices, df_1d, daily_bearish)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if daily trend data not available
        if np.isnan(daily_bullish_aligned[i]) or np.isnan(daily_bearish_aligned[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits
        if position == 1:  # long position
            # Exit: price falls below Donchian low or daily turn bearish
            if (close[i] < donchian_low[i] or 
                daily_bearish_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price rises above Donchian high or daily turn bullish
            if (close[i] > donchian_high[i] or 
                daily_bullish_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with daily trend filter and volume confirmation
            # Long: price breaks above Donchian high with volume > average during bullish day
            if (close[i] > donchian_high[i] and 
                volume[i] > volume_ma[i] and 
                daily_bullish_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low with volume > average during bearish day
            elif (close[i] < donchian_low[i] and 
                  volume[i] > volume_ma[i] and 
                  daily_bearish_aligned[i]):
                signals[i] = -0.25
                position = -1
    
    return signals