#!/usr/bin/env python3
"""
Hypothesis: 6h Donchian(20) breakout with 1d weekly pivot bias and volume confirmation.
- Primary timeframe: 6h for execution, HTF: 1d for weekly pivot bias.
- Weekly pivot bias: if weekly close > weekly open → bullish bias (long breakouts only)
                      if weekly close < weekly open → bearish bias (short breakdowns only)
- Entry: Long when price breaks above 6h Donchian upper (20) with volume spike and bullish weekly bias.
         Short when price breaks below 6h Donchian lower (20) with volume spike and bearish weekly bias.
- Exit: When price returns to the 6h Donchian midpoint (mean reversion edge).
- Works in bull via buying breakouts in uptrend bias, in bear via selling breakdowns in downtrend bias.
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_donchian(high, low, window=20):
    """Calculate Donchian channel upper and lower bands"""
    upper = pd.Series(high).rolling(window=window, min_periods=window).max().values
    lower = pd.Series(low).rolling(window=window, min_periods=window).min().values
    return upper, lower

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for weekly pivot bias
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 7:
        return np.zeros(n)
    
    # Calculate weekly OHLC from daily data (weekly bias)
    # Group by week: Monday=0, Sunday=6
    df_1d = df_1d.copy()
    df_1d['week_start'] = df_1d.index - pd.to_timedelta(df_1d.index.weekday, unit='D')
    weekly = df_1d.groupby('week_start').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last'
    }).reset_index()
    
    # Calculate weekly bias: 1 if bullish (close > open), -1 if bearish (close < open), 0 if doji
    weekly_bias_raw = np.where(weekly['close'] > weekly['open'], 1,
                       np.where(weekly['close'] < weekly['open'], -1, 0))
    
    # Create arrays aligned to daily data
    weekly_bias_d1 = np.repeat(weekly_bias_raw, [len(g) for _, g in df_1d.groupby(df_1d.index - pd.to_timedelta(df_1d.index.weekday, unit='D'))])
    
    # Align weekly bias to 6h timeframe
    weekly_bias_aligned = align_htf_to_ltf(prices, df_1d, weekly_bias_d1)
    
    # Calculate 6h Donchian channels
    upper, lower = calculate_donchian(high, low, window=20)
    midpoint = (upper + lower) / 2
    
    # Volume confirmation: current volume > 2.0 * 20-period volume MA (on 6h)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 20)  # Need enough for Donchian and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(midpoint[i]) or np.isnan(weekly_bias_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Check for breakout signals with volume spike and weekly bias filter
            if volume_spike[i]:
                # Bullish breakout: price > upper and bullish weekly bias
                if close[i] > upper[i] and weekly_bias_aligned[i] > 0:
                    signals[i] = 0.25
                    position = 1
                # Bearish breakdown: price < lower and bearish weekly bias
                elif close[i] < lower[i] and weekly_bias_aligned[i] < 0:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price returns to midpoint (mean reversion) or stoploss
            if close[i] <= midpoint[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price returns to midpoint (mean reversion) or stoploss
            if close[i] >= midpoint[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_Breakout_1dWeeklyPivotBias_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0