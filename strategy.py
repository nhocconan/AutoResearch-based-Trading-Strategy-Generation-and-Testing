#!/usr/bin/env python3
"""
Hypothesis: 6h Donchian(20) breakout with weekly pivot direction and volume confirmation.
- Uses 6h timeframe (primary) and 1w HTF for weekly pivot-based trend alignment (novel combination)
- Weekly pivot levels (PP, R1, S1) calculated from prior completed week's OHLC
- Long when price breaks above Donchian upper AND price > weekly PP (bullish bias) AND volume > 1.8 * volume MA(20)
- Short when price breaks below Donchian lower AND price < weekly PP (bearish bias) AND volume > 1.8 * volume MA(20)
- Exit when price crosses the weekly PP (mean reversion to pivot point)
- Discrete signal size: 0.25 to minimize fee churn
- Target: 50-150 total trades over 4 years (12-37/year) as per 6h timeframe recommendation
- Works in both bull/bear: weekly pivot provides structural bias, Donchian breakouts capture momentum with volume confirmation
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 6h Donchian levels from previous completed 6h bar's OHLC (lookback=20)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    
    lookback = 20
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(lookback, n):
        start_idx = i - lookback
        end_idx = i  # [start_idx, end_idx-1] = [i-lookback, i-1]
        if start_idx >= 0 and not (np.isnan(prev_high[start_idx:end_idx]).any() or np.isnan(prev_low[start_idx:end_idx]).any()):
            upper[i] = np.max(prev_high[start_idx:end_idx])
            lower[i] = np.min(prev_low[start_idx:end_idx])
    
    # Calculate weekly pivot points (PP, R1, S1) from prior completed week's OHLC
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:  # Need at least one weekly bar
        return np.zeros(n)
    
    # Weekly OHLC
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    # Weekly pivot: PP = (H + L + C) / 3
    weekly_pp = (weekly_high + weekly_low + weekly_close) / 3.0
    # Align weekly PP to 6h timeframe (wait for weekly bar to close)
    weekly_pp_aligned = align_htf_to_ltf(prices, df_1w, weekly_pp)
    
    # Volume confirmation: current volume > 1.8 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.8 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(lookback, 20)  # Need Donchian lookback and volume MA(20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(weekly_pp_aligned[i]) or 
            np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian upper AND price > weekly PP (bullish bias) AND volume confirmation
            if close[i] > upper[i] and close[i] > weekly_pp_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower AND price < weekly PP (bearish bias) AND volume confirmation
            elif close[i] < lower[i] and close[i] < weekly_pp_aligned[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below weekly PP (mean reversion to pivot)
            if close[i] < weekly_pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above weekly PP (mean reversion to pivot)
            if close[i] > weekly_pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_WeeklyPP_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0