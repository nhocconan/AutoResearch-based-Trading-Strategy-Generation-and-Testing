#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 12h trend filter and volume confirmation.
# Long when price breaks above upper Donchian channel during bullish 12h trend with volume > 1.2x 20-period average.
# Short when price breaks below lower Donchian channel during bearish 12h trend with volume confirmation.
# Uses 12h trend filter to avoid counter-trend trades. Donchian channels provide clear breakout points.
# Target: 50-150 total trades over 4 years (12-37/year) to stay within optimal range.

name = "6h_donchian20_12h_trend_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channel (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    upper = high_series.rolling(window=20, min_periods=20).max().values
    lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # 12h trend filter: bullish/bearish based on close vs open
    df_12h = get_htf_data(prices, '12h')
    open_12h = df_12h['open'].values
    close_12h = df_12h['close'].values
    bullish_12h = close_12h > open_12h  # True for bullish 12h bar
    bearish_12h = close_12h < open_12h   # True for bearish 12h bar
    bullish_12h_aligned = align_htf_to_ltf(prices, df_12h, bullish_12h)
    bearish_12h_aligned = align_htf_to_ltf(prices, df_12h, bearish_12h)
    
    # Volume filter: current volume > 1.2x 20-period average
    volume_series = pd.Series(volume)
    vol_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if 12h trend data not available
        if np.isnan(bullish_12h_aligned[i]) or np.isnan(bearish_12h_aligned[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 1.2
        
        # Check exits
        if position == 1:  # long position
            # Exit: price drops below lower Donchian or 12h turn bearish
            if (low[i] <= lower[i] or 
                bearish_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price rises above upper Donchian or 12h turn bullish
            if (high[i] >= upper[i] or 
                bullish_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation and 12h trend filter
            if volume_filter:
                # Long: break above upper Donchian during bullish 12h trend
                if (high[i] > upper[i] and 
                    bullish_12h_aligned[i]):
                    signals[i] = 0.25
                    position = 1
                # Short: break below lower Donchian during bearish 12h trend
                elif (low[i] < lower[i] and 
                      bearish_12h_aligned[i]):
                    signals[i] = -0.25
                    position = -1
    
    return signals