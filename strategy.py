#!/usr/bin/env python3
# 6h_donchian_breakout_1w_pivot_volume_v1
# Hypothesis: 6h Donchian(20) breakouts in direction of 1w Camarilla pivot bias (above/below pivot) with volume confirmation.
# Works in bull/bear: 1w pivot defines regime (bull/bear/range), Donchian captures breakouts, volume filters false signals.
# Target: 12-37 trades/year (~50-150 total over 4 years).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_donchian_breakout_1w_pivot_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d HTF data for 1w Camarilla pivot calculation (using 1d to build 1w)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # Calculate weekly Camarilla pivot points from prior week's OHLC
    # We need to resample 1d to weekly manually since we only have daily data
    # Build weekly OHLC from daily data
    df_1d = df_1d.copy()
    df_1d['open_time'] = pd.to_datetime(df_1d['open_time'])
    df_1d.set_index('open_time', inplace=True)
    # Resample to weekly: Friday end of week
    df_weekly = df_1d.resample('W-FRI').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last'
    }).dropna()
    
    if len(df_weekly) < 1:
        return np.zeros(n)
    
    # Calculate Camarilla levels for each weekly bar
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    close_weekly = df_weekly['close'].values
    
    # Camarilla formulas:
    # Pivot = (High + Low + Close) / 3
    # R4 = Close + ((High - Low) * 1.1 / 2)
    # R3 = Close + ((High - Low) * 1.1 / 4)
    # S3 = Close - ((High - Low) * 1.1 / 4)
    # S4 = Close - ((High - Low) * 1.1 / 2)
    pivot_weekly = (high_weekly + low_weekly + close_weekly) / 3.0
    r4 = close_weekly + ((high_weekly - low_weekly) * 1.1 / 2.0)
    r3 = close_weekly + ((high_weekly - low_weekly) * 1.1 / 4.0)
    s3 = close_weekly - ((high_weekly - low_weekly) * 1.1 / 4.0)
    s4 = close_weekly - ((high_weekly - low_weekly) * 1.1 / 2.0)
    
    # Align weekly Camarilla levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_weekly, pivot_weekly)
    r4_aligned = align_htf_to_ltf(prices, df_weekly, r4)
    r3_aligned = align_htf_to_ltf(prices, df_weekly, r3)
    s3_aligned = align_htf_to_ltf(prices, df_weekly, s3)
    s4_aligned = align_htf_to_ltf(prices, df_weekly, s4)
    
    # 6h Donchian channel (20-period)
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(lookback, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(pivot_aligned[i]) or
            np.isnan(r4_aligned[i]) or np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price breaks below S3 (mean reversion) or breaks above R4 (take profit)
            if close[i] < s3_aligned[i] or close[i] > r4_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above R3 (mean reversion) or breaks below S4 (take profit)
            if close[i] > r3_aligned[i] or close[i] < s4_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Volume confirmation
            volume_confirmed = volume[i] > 1.5 * volume_ma[i]
            
            if volume_confirmed:
                # Long breakout: price breaks above R4 with bullish bias (above weekly pivot)
                if close[i] > highest_high[i] and close[i] > r4_aligned[i] and close[i] > pivot_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short breakout: price breaks below S4 with bearish bias (below weekly pivot)
                elif close[i] < lowest_low[i] and close[i] < s4_aligned[i] and close[i] < pivot_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals