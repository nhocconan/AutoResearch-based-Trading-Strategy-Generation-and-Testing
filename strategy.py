#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout + 12h EMA50 trend filter + volume spike confirmation.
- Primary timeframe: 4h for execution, HTF: 12h for EMA trend.
- Entry: Long when price breaks above Donchian(20) high with volume spike and price > 12h EMA50 (uptrend).
         Short when price breaks below Donchian(20) low with volume spike and price < 12h EMA50 (downtrend).
- Exit: Opposite Donchian breakout or when price closes back inside the Donchian channel.
- Uses discrete signal size 0.30 to balance drawdown control and fee efficiency.
- Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe.
- Works in bull via buying breakouts in uptrend, in bear via selling breakdowns in downtrend.
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
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Donchian(20) channels on 4h
    lookback = 20
    donchian_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    donchian_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume confirmation: current volume > 2.0 * 20-period volume MA (on 4h)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20, 20)  # EMA50 + Donchian20 + volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Check for Donchian breakout with volume spike and trend filter
            if volume_spike[i]:
                # Bullish breakout: price > Donchian high and above 12h EMA50
                if close[i] > donchian_high[i] and close[i] > ema_50_12h_aligned[i]:
                    signals[i] = 0.30
                    position = 1
                # Bearish breakout: price < Donchian low and below 12h EMA50
                elif close[i] < donchian_low[i] and close[i] < ema_50_12h_aligned[i]:
                    signals[i] = -0.30
                    position = -1
        elif position == 1:
            # Long exit: price closes below Donchian low or opposite breakout
            if close[i] < donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Short exit: price closes above Donchian high or opposite breakout
            if close[i] > donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

name = "4h_Donchian20_Breakout_12hEMA50_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0