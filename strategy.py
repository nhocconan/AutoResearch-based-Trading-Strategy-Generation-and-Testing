#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-hour strategy using 4h Donchian(20) for direction, 1d trend filter, and volume confirmation.
# Long when price breaks above 4h upper Donchian during bullish day (close > open) with volume > 1.5x 20-period average.
# Short when price breaks below 4h lower Donchian during bearish day (close < open) with volume confirmation.
# Uses 4h Donchian for trend direction (reduces whipsaw) and 1d trend to avoid counter-trend trades.
# Volume filter ensures momentum behind breakouts. Target: 60-150 total trades over 4 years (15-37/year).

name = "1h_donchian20_1d_trend_vol_v2"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 4h Donchian channel (20-period)
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    high_series = pd.Series(high_4h)
    low_series = pd.Series(low_4h)
    upper_4h = high_series.rolling(window=20, min_periods=20).max().values
    lower_4h = low_series.rolling(window=20, min_periods=20).min().values
    upper_4h_aligned = align_htf_to_ltf(prices, df_4h, upper_4h)
    lower_4h_aligned = align_htf_to_ltf(prices, df_4h, lower_4h)
    
    # Daily trend filter: bullish/bearish day based on close vs open
    df_1d = get_htf_data(prices, '1d')
    daily_open = df_1d['open'].values
    daily_close = df_1d['close'].values
    daily_bullish = daily_close > daily_open  # True for bullish day
    daily_bearish = daily_close < daily_open   # True for bearish day
    daily_bullish_aligned = align_htf_to_ltf(prices, df_1d, daily_bullish)
    daily_bearish_aligned = align_htf_to_ltf(prices, df_1d, daily_bearish)
    
    # Volume filter: current volume > 1.5x 20-period average
    volume_series = pd.Series(volume)
    vol_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if 4h or daily trend data not available
        if np.isnan(upper_4h_aligned[i]) or np.isnan(lower_4h_aligned[i]) or \
           np.isnan(daily_bullish_aligned[i]) or np.isnan(daily_bearish_aligned[i]):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 1.5
        
        # Check exits
        if position == 1:  # long position
            # Exit: price drops below 4h lower Donchian or daily turn bearish
            if (low[i] <= lower_4h_aligned[i] or 
                daily_bearish_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            # Exit: price rises above 4h upper Donchian or daily turn bullish
            if (high[i] >= upper_4h_aligned[i] or 
                daily_bullish_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
        else:
            # Look for entries with volume confirmation and daily trend filter
            if volume_filter:
                # Long: break above 4h upper Donchian during bullish day
                if (high[i] > upper_4h_aligned[i] and 
                    daily_bullish_aligned[i]):
                    signals[i] = 0.20
                    position = 1
                # Short: break below 4h lower Donchian during bearish day
                elif (low[i] < lower_4h_aligned[i] and 
                      daily_bearish_aligned[i]):
                    signals[i] = -0.20
                    position = -1
    
    return signals