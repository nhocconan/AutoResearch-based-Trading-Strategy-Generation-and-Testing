#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout + 1w EMA50 trend filter + Volume confirmation.
- Primary timeframe: 1d for execution, HTF: 1w for EMA50 trend filter.
- Donchian breakout: Long when price > 20-day high, Short when price < 20-day low.
- Trend filter: 1w EMA50 slope > 0 for long bias, < 0 for short bias.
- Volume confirmation: current volume > 1.5x 20-day volume MA to ensure momentum.
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 30-100 total trades over 4 years (7-25/year) for 1d timeframe.
- Works in bull via buying breakouts in uptrend, in bear via selling breakdowns in downtrend.
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
    
    # Calculate 1d Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d volume MA (20-period)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1w EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 20) + 1  # Need Donchian(20), volume MA(20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(ema_50_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.5 * 20-day volume MA
        volume_confirmed = volume[i] > (1.5 * volume_ma[i])
        
        if position == 0:
            # Long: price breaks above 20-day high AND 1w EMA50 rising AND volume confirmed
            if close[i] > highest_high[i] and ema_50_1w_aligned[i] > ema_50_1w_aligned[i-1] and volume_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 20-day low AND 1w EMA50 falling AND volume confirmed
            elif close[i] < lowest_low[i] and ema_50_1w_aligned[i] < ema_50_1w_aligned[i-1] and volume_confirmed:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price returns to 20-day low or opposite signal
            if close[i] < lowest_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price returns to 20-day high or opposite signal
            if close[i] > highest_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_1wEMA50_Trend_VolumeConfirm_v1"
timeframe = "1d"
leverage = 1.0