#!/usr/bin/env python3
"""
Hypothesis: 6h Donchian(20) breakout with 12h EMA50 trend filter and volume spike confirmation.
- Primary timeframe: 6h for execution, HTF: 12h for EMA trend and Donchian levels.
- Donchian levels calculated from previous 12h OHLC (highest high/lowest low over 20 periods).
- Entry: Long when price breaks above upper Donchian with volume spike and close > 12h EMA50 (uptrend).
         Short when price breaks below lower Donchian with volume spike and close < 12h EMA50 (downtrend).
- Exit: When price returns to the midpoint of the Donchian channel (mean reversion edge).
- Works in bull via buying breakouts in uptrend, in bear via selling breakdowns in downtrend.
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_donchian(high, low, window=20):
    """Calculate Donchian channel levels for given high/low arrays"""
    if len(high) < window:
        return np.full_like(high, np.nan), np.full_like(high, np.nan), np.full_like(high, np.nan)
    upper = pd.Series(high).rolling(window=window, min_periods=window).max().values
    lower = pd.Series(low).rolling(window=window, min_periods=window).min().values
    middle = (upper + lower) / 2
    return upper, lower, middle

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Donchian levels and EMA trend
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:  # Need enough for Donchian(20) and EMA50
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    ema_50 = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate Donchian levels for each 12h bar (using previous 20 periods)
    upper_12h, lower_12h, middle_12h = calculate_donchian(
        df_12h['high'].values, 
        df_12h['low'].values, 
        window=20
    )
    
    # Align 12h indicators to 6h
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50)
    upper_aligned = align_htf_to_ltf(prices, df_12h, upper_12h)
    lower_aligned = align_htf_to_ltf(prices, df_12h, lower_12h)
    middle_aligned = align_htf_to_ltf(prices, df_12h, middle_12h)
    
    # Volume confirmation: current volume > 2.0 * 20-period volume MA (on 6h)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # Need enough 12h bars for EMA50 and Donchian
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_aligned[i]) or np.isnan(upper_aligned[i]) or 
            np.isnan(lower_aligned[i]) or np.isnan(middle_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Check for breakout signals with volume spike and trend filter
            if volume_spike[i]:
                # Bullish breakout: price > upper Donchian and close > EMA50
                if close[i] > upper_aligned[i] and close[i] > ema_50_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Bearish breakdown: price < lower Donchian and close < EMA50
                elif close[i] < lower_aligned[i] and close[i] < ema_50_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price returns to middle of Donchian channel (mean reversion)
            if close[i] <= middle_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price returns to middle of Donchian channel (mean reversion)
            if close[i] >= middle_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_Breakout_12hEMA50_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0