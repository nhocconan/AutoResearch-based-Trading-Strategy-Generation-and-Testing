#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation.
- Primary timeframe: 4h, HTF: 1d for EMA50 trend alignment.
- Donchian breakout: long when price > 20-period high, short when price < 20-period low.
- Trend filter: only long when 4h close > 1d EMA50, only short when 4h close < 1d EMA50.
- Volume confirmation: current 4h volume > 2.0 * 20-period 4h volume MA.
- Discrete signal size: 0.25 to minimize fee churn and control drawdown.
- Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe.
- Exit: opposite Donchian breakout (long exit when price < 10-period low, short exit when price > 10-period high).
- Works in bull via trend-aligned breakouts, in bear via short breakouts with trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Donchian channels (20-period for entry, 10-period for exit)
    high_roll = pd.Series(high)
    low_roll = pd.Series(low)
    donchian_high = high_roll.rolling(window=20, min_periods=20).max().values
    donchian_low = low_roll.rolling(window=20, min_periods=20).min().values
    donchian_high_exit = high_roll.rolling(window=10, min_periods=10).max().values
    donchian_low_exit = low_roll.rolling(window=10, min_periods=10).min().values
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: current volume > 2.0 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # Need 1d EMA50 and 20-period Donchian
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price > Donchian high AND uptrend AND volume spike
            if close[i] > donchian_high[i] and close[i] > ema_50_1d_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: price < Donchian low AND downtrend AND volume spike
            elif close[i] < donchian_low[i] and close[i] < ema_50_1d_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price < 10-period Donchian low
            if close[i] < donchian_low_exit[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price > 10-period Donchian high
            if close[i] > donchian_high_exit[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_1dEMA50_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0