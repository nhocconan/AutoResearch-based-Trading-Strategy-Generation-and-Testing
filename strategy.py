#!/usr/bin/env python3
"""
12h_Donchian20_Breakout_1dTrend_VolumeFilter
Hypothesis: Donchian channel breakout on 12h timeframe with 1d EMA50 trend filter and volume confirmation (>1.3x 20-period MA).
Long when price breaks above upper Donchian(20) with 1d uptrend and volume spike.
Short when price breaks below lower Donchian(20) with 1d downtrend and volume filter.
Exit on opposite Donchian breakout or trend reversal.
Uses discrete position sizing (0.25) to minimize fee churn.
Designed to capture medium-term trends in both bull and bear markets by following the 1d trend.
Target: 12-37 trades/year (50-150 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # 1d EMA50 trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    uptrend_1d = close > ema_50_1d_aligned
    downtrend_1d = close < ema_50_1d_aligned
    
    # 12h Donchian channel (20-period)
    lookback = 20
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(lookback - 1, n):
        upper[i] = np.max(high[i - lookback + 1:i + 1])
        lower[i] = np.min(low[i - lookback + 1:i + 1])
    
    # Volume confirmation: volume > 1.3x 20-period MA
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 50 for 1d EMA + 20 for Donchian + 20 for volume MA)
    start_idx = 70
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(upper[i]) or 
            np.isnan(lower[i]) or np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:
            # Long: price breaks above upper Donchian with 1d uptrend and volume spike
            if close[i] > upper[i] and uptrend_1d[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian with 1d downtrend and volume spike
            elif close[i] < lower[i] and downtrend_1d[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price breaks below lower Donchian OR 1d trend changes to downtrend
            if close[i] < lower[i] or not uptrend_1d[i]:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price breaks above upper Donchian OR 1d trend changes to uptrend
            if close[i] > upper[i] or not downtrend_1d[i]:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Donchian20_Breakout_1dTrend_VolumeFilter"
timeframe = "12h"
leverage = 1.0