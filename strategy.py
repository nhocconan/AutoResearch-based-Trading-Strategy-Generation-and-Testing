#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h 1234 Reversal pattern with 1d volume confirmation and ADX trend filter.
# Long when: price forms a 1234 bottom (low1 > low2 > low3 > low4), 1d volume > 1.2x 20-period average, ADX(14) > 20.
# Short when: price forms a 1234 top (high1 < high2 < high3 < high4), 1d volume > 1.2x 20-period average, ADX(14) > 20.
# Exit when price closes back inside the 1234 pattern range.
# Uses 4h timeframe with 1d context for higher timeframe confirmation.
# Target: 100-200 total trades over 4 years (25-50/year) with controlled frequency to avoid fee drag.

name = "4h_1234_Reversal_1dVolume_ADX"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Daily data for volume and ADX
    df_d = get_htf_data(prices, '1d')
    if len(df_d) < 2:
        return np.zeros(n)
    
    # 1234 pattern detection on 4h data
    # For longs: looking for higher lows (bullish reversal)
    # low1 (4 bars ago) > low2 (3 bars ago) > low3 (2 bars ago) > low4 (1 bar ago)
    low1 = np.roll(low, 4)
    low2 = np.roll(low, 3)
    low3 = np.roll(low, 2)
    low4 = np.roll(low, 1)
    
    bullish_1234 = (low1 > low2) & (low2 > low3) & (low3 > low4)
    # For shorts: looking for lower highs (bearish reversal)
    # high1 (4 bars ago) < high2 (3 bars ago) < high3 (2 bars ago) < high4 (1 bar ago)
    high1 = np.roll(high, 4)
    high2 = np.roll(high, 3)
    high3 = np.roll(high, 2)
    high4 = np.roll(high, 1)
    
    bearish_1234 = (high1 < high2) & (high2 < high3) & (high3 < high4)
    
    # Daily volume filter: current volume > 1.2x 20-period average
    volume_d = df_d['volume'].values
    vol_ma20_d = pd.Series(volume_d).rolling(window=20, min_periods=20).mean().values
    volume_filter_d = volume_d > (1.2 * vol_ma20_d)
    volume_filter = align_htf_to_ltf(prices, df_d, volume_filter_d)
    
    # Daily ADX(14) for trend strength
    high_d = df_d['high'].values
    low_d = df_d['low'].values
    close_d = df_d['close'].values
    
    # True Range
    tr1 = high_d - low_d
    tr2 = np.abs(high_d - np.roll(close_d, 1))
    tr3 = np.abs(low_d - np.roll(close_d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = high_d[0] - low_d[0]  # First TR
    
    # Directional Movement
    plus_dm = np.where((high_d - np.roll(high_d, 1)) > (np.roll(low_d, 1) - low_d), 
                       np.maximum(high_d - np.roll(high_d, 1), 0), 0)
    minus_dm = np.where((np.roll(low_d, 1) - low_d) > (high_d - np.roll(high_d, 1)), 
                        np.maximum(np.roll(low_d, 1) - low_d, 0), 0)
    plus_dm[0] = 0
    minus_dm[0] = 0
    
    # Smoothed values
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_dm_smooth = pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / atr
    minus_di = 100 * minus_dm_smooth / atr
    
    # ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    adx[np.isnan(adx)] = 0
    
    # Align ADX to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_d, adx)
    
    # Trend filter: ADX > 20 (weaker trend filter to allow more trades in ranging markets)
    trend_filter = adx_aligned > 20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 5  # Need at least 5 bars for 1234 pattern
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(bullish_1234[i]) or np.isnan(bearish_1234[i]) or 
            np.isnan(volume_filter[i]) or np.isnan(trend_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: bullish 1234 pattern, volume filter, trending/ranging market
            long_cond = bullish_1234[i] and volume_filter[i] and trend_filter[i]
            # Short conditions: bearish 1234 pattern, volume filter, trending/ranging market
            short_cond = bearish_1234[i] and volume_filter[i] and trend_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price closes below the lowest point of the 1234 pattern
            exit_level = min(low[i-3], low[i-2], low[i-1], low[i])
            if close[i] < exit_level:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price closes above the highest point of the 1234 pattern
            exit_level = max(high[i-3], high[i-2], high[i-1], high[i])
            if close[i] > exit_level:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals