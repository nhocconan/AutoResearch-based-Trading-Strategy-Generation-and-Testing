#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation.
- Primary timeframe: 12h for execution, HTF: 1w for EMA trend and Donchian levels.
- Donchian channel calculated from previous 1w OHLC (highest high/lowest low over 20 periods).
- Entry: Long when price breaks above upper Donchian with volume spike and close > 1w EMA50.
         Short when price breaks below lower Donchian with volume spike and close < 1w EMA50.
- Exit: When price returns to the midpoint of the Donchian channel (mean reversion edge).
- Works in bull via buying breakouts in uptrend, in bear via selling breakdowns in downtrend.
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for Donchian levels and EMA trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    ema_50 = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate Donchian levels for each 1w bar (highest high/lowest low over 20 periods)
    upper = np.full(len(df_1w), np.nan)
    lower = np.full(len(df_1w), np.nan)
    midpoint = np.full(len(df_1w), np.nan)
    
    for i in range(len(df_1w)):
        if i >= 19:  # Need 20 periods for Donchian (0-indexed: 0-19 = 20 bars)
            window_high = df_1w['high'].iloc[i-19:i+1].values
            window_low = df_1w['low'].iloc[i-19:i+1].values
            upper[i] = np.max(window_high)
            lower[i] = np.min(window_low)
            midpoint[i] = (upper[i] + lower[i]) / 2.0
    
    # Align 1w indicators to 12h
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    upper_aligned = align_htf_to_ltf(prices, df_1w, upper)
    lower_aligned = align_htf_to_ltf(prices, df_1w, lower)
    midpoint_aligned = align_htf_to_ltf(prices, df_1w, midpoint)
    
    # Volume confirmation: current volume > 2.0 * 20-period volume MA (on 12h)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # Need enough 1w bars for EMA50 and Donchian
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_aligned[i]) or np.isnan(upper_aligned[i]) or 
            np.isnan(lower_aligned[i]) or np.isnan(midpoint_aligned[i]) or 
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
            # Long exit: price returns to midpoint (mean reversion) or stoploss
            if close[i] <= midpoint_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price returns to midpoint (mean reversion) or stoploss
            if close[i] >= midpoint_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_Breakout_1wEMA50_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0