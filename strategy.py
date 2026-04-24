#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume spike confirmation.
- Uses 1d timeframe (primary) and 1w HTF for EMA50 trend alignment
- Donchian breakout: long when price > highest high of last 20 days, short when price < lowest low of last 20 days
- Trend filter: only long when price > 1w EMA50, only short when price < 1w EMA50
- Volume confirmation: current volume > 2.0 * 20-period volume MA to avoid low-volume false signals
- Exit: reverse signal or when price crosses 10-day EMA (mean reversion)
- Discrete signal size: 0.25 to balance return and fee drag
- Target: 50-100 total trades over 4 years (12-25/year) for 1d timeframe
- Works in both bull/bear: trend filter avoids counter-trend trades, Donchian captures breakouts in all regimes
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Extract price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1w EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Donchian(20) channels
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume confirmation: current volume > 2.0 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * volume_ma)
    
    # Trend filter: price above/below 1w EMA50
    uptrend = close > ema_50_1w_aligned
    downtrend = close < ema_50_1w_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # Need 1w EMA50 and Donchian lookback
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian upper band AND uptrend AND volume confirmation
            if close[i] > highest_high[i] and uptrend[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower band AND downtrend AND volume confirmation
            elif close[i] < lowest_low[i] and downtrend[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below 10-day EMA (mean reversion) or reverse signal
            ema_10 = pd.Series(close).ewm(span=10, adjust=False, min_periods=10).mean().values
            if close[i] < ema_10[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above 10-day EMA (mean reversion) or reverse signal
            ema_10 = pd.Series(close).ewm(span=10, adjust=False, min_periods=10).mean().values
            if close[i] > ema_10[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_1wEMA50_VolumeConfirm_v1"
timeframe = "1d"
leverage = 1.0