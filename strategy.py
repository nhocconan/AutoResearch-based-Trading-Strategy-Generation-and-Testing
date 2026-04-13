#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Donchian(20) breakout with 1d weekly pivot direction filter and volume confirmation
    # Long when price > upper Donchian + price above weekly pivot + volume > 1.5x 20-period average
    # Short when price < lower Donchian + price below weekly pivot + volume > 1.5x 20-period average
    # Exit when price crosses middle Donchian
    # Discrete position sizing: 0.25 to limit drawdown and reduce fee churn
    # Target: 50-150 total trades over 4 years (~12-38/year) to avoid fee drag
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 6h Donchian channels (20-period)
    high_6h = high
    low_6h = low
    close_6h = close
    
    # Upper channel: highest high of last 20 periods
    upper = pd.Series(high_6h).rolling(window=20, min_periods=20).max().values
    # Lower channel: lowest low of last 20 periods
    lower = pd.Series(low_6h).rolling(window=20, min_periods=20).min().values
    # Middle channel: average of upper and lower
    middle = (upper + lower) / 2
    
    # Calculate 1d weekly pivot points (using prior week's high, low, close)
    # For simplicity, we'll use prior day's OHLC as proxy for weekly (more frequent signals)
    # In practice, would use actual weekly data, but daily is more available and still effective
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    open_1d = df_1d['open'].values
    
    # Prior day's typical price as pivot point
    typical_price = (high_1d + low_1d + close_1d) / 3
    # Weekly pivot approximation: using prior day's typical price
    pivot = typical_price
    # Resistance 1 and Support 1
    r1 = 2 * pivot - low_1d
    s1 = 2 * pivot - high_1d
    
    # Align 1d pivot levels to 6h
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Calculate 1d volume average (20-period) with min_periods
    volume_1d = df_1d['volume'].values
    volume_series = pd.Series(volume_1d)
    vol_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(middle[i]) or 
            np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(vol_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current 1d volume > 1.5 * 20-period average (volume expansion)
        vol_1d_current = df_1d['volume'].values
        vol_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_1d_current)
        volume_expansion = vol_1d_aligned[i] > 1.5 * vol_ma_aligned[i]
        
        # Breakout conditions with pivot filter
        bullish_breakout = (close[i] > upper[i] and 
                           close[i] > pivot_aligned[i] and 
                           volume_expansion)
        bearish_breakout = (close[i] < lower[i] and 
                           close[i] < pivot_aligned[i] and 
                           volume_expansion)
        
        # Exit condition: price returns to middle Donchian
        long_exit = close[i] < middle[i]
        short_exit = close[i] > middle[i]
        
        if bullish_breakout and position != 1:
            position = 1
            signals[i] = 0.25
        elif bearish_breakout and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_1d_donchian_weekly_pivot_volume_v2"
timeframe = "6h"
leverage = 1.0