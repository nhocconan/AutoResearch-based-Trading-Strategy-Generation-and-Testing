#!/usr/bin/env python3
"""
6h_Donchian_Breakout_1dTrend_Volume
Hypothesis: 6-hour Donchian channel breakout (20-period) in the direction of 1-day trend (EMA50) with volume confirmation (volume > 1.5x 20-period average). This strategy captures momentum bursts aligned with the daily trend, using the Donchian channel as a dynamic breakout filter. Designed for 6h to limit trades (target 50-150 over 4 years) and avoid fee drag. Works in bull markets via breakout continuation and in bear markets via short breakdowns with trend filter.
"""

name = "6h_Donchian_Breakout_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # Calculate daily EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate volume average (20-period) for volume confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Donchian channel (20-period) on 6h data
    # Upper band: highest high over past 20 periods
    # Lower band: lowest low over past 20 periods
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i]) or 
            np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # LONG: price breaks above Donchian upper + volume + price above daily EMA50 (uptrend)
            if close[i] > donchian_upper[i] and vol_confirmed and close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: price breaks below Donchian lower + volume + price below daily EMA50 (downtrend)
            elif close[i] < donchian_lower[i] and vol_confirmed and close[i] < ema_50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price breaks below Donchian lower or trend changes (price below EMA50)
            if close[i] < donchian_lower[i] or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price breaks above Donchian upper or trend changes (price above EMA50)
            if close[i] > donchian_upper[i] or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals