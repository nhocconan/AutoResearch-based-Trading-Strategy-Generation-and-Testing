#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter and volume spike confirmation.
- Primary timeframe: 4h for execution, HTF: 12h for EMA trend.
- Donchian channel calculated from 20-period high/low on 4h data.
- Entry: Long when price breaks above upper Donchian with volume spike and close > 12h EMA50 (uptrend).
         Short when price breaks below lower Donchian with volume spike and close < 12h EMA50 (downtrend).
- Exit: When price returns to the opposite Donchian level (mean reversion) or ATR-based stoploss.
- Works in bull via buying breakouts in uptrend, in bear via selling breakdowns in downtrend.
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA trend
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    ema_50 = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50)
    
    # Calculate Donchian channels on 4h (20-period)
    lookback = 20
    upper = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lower = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume confirmation: current volume > 2.0 * 20-period volume MA (on 4h)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * volume_ma)
    
    # ATR for stoploss (optional, using close-based exit instead)
    # We'll use mean reversion to opposite band as exit
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(lookback, 20, 50)  # Need enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_aligned[i]) or np.isnan(upper[i]) or 
            np.isnan(lower[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Check for breakout signals with volume spike and trend filter
            if volume_spike[i]:
                # Bullish breakout: price > upper Donchian and close > EMA50
                if close[i] > upper[i] and close[i] > ema_50_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Bearish breakdown: price < lower Donchian and close < EMA50
                elif close[i] < lower[i] and close[i] < ema_50_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price returns to lower Donchian (mean reversion)
            if close[i] <= lower[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price returns to upper Donchian (mean reversion)
            if close[i] >= upper[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Breakout_12hEMA50_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0