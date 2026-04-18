#!/usr/bin/env python3
"""
4h_DonchianBreakout_VolumeTrend
4h strategy using Donchian(20) breakout with 1d EMA trend filter and volume confirmation.
- Long: Close breaks above Donchian Upper (20-period high) + 1d EMA50 > EMA200 + Volume > 1.5x 20-period average
- Short: Close breaks below Donchian Lower (20-period low) + 1d EMA50 < EMA200 + Volume > 1.5x 20-period average
- Exit: Opposite breakout
Designed for ~20-40 trades/year per symbol (80-160 total over 4 years)
Works in bull markets (breakout continuation) and bear markets (breakdown continuation)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    close_1d = df_1d['close'].values
    
    # 1d EMA50 and EMA200 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Donchian Channels (20-period)
    lookback = 20
    donchian_upper = np.full(n, np.nan)
    donchian_lower = np.full(n, np.nan)
    
    for i in range(n):
        if i >= lookback - 1:
            donchian_upper[i] = np.max(high[i - lookback + 1:i + 1])
            donchian_lower[i] = np.min(low[i - lookback + 1:i + 1])
    
    # Volume average (20-period)
    vol_avg = np.full(n, np.nan)
    for i in range(n):
        if i >= lookback - 1:
            vol_avg[i] = np.mean(volume[i - lookback + 1:i + 1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, lookback - 1)  # need enough for EMA200 and Donchian
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(vol_avg[i]) or np.isnan(ema_50_aligned[i]) or np.isnan(ema_200_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend conditions
        uptrend = ema_50_aligned[i] > ema_200_aligned[i]
        downtrend = ema_50_aligned[i] < ema_200_aligned[i]
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_avg[i]
        
        # Breakout conditions
        breakout_up = close[i] > donchian_upper[i]
        breakdown_down = close[i] < donchian_lower[i]
        
        if position == 0:
            # Long: uptrend + breakout above Donchian Upper + volume confirmation
            if uptrend and breakout_up and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: downtrend + breakdown below Donchian Lower + volume confirmation
            elif downtrend and breakdown_down and vol_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: breakdown below Donchian Lower
            if breakdown_down:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: breakout above Donchian Upper
            if breakout_up:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_DonchianBreakout_VolumeTrend"
timeframe = "4h"
leverage = 1.0