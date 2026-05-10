#!/usr/bin/env python3
"""
4h_Donchian_Breakout_Volume_Trend_1w
Hypothesis: Donchian(20) breakout with weekly trend filter and volume confirmation.
Works in bull markets by catching breakouts above the 20-period high and in bear markets
by catching breakdowns below the 20-period low. Weekly trend ensures alignment with higher timeframe momentum.
Volume confirmation filters out low-conviction breakouts. Target: 20-40 trades/year per symbol.
"""

name = "4h_Donchian_Breakout_Volume_Trend_1w"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Convert to Series for indicator calculations
    close_s = pd.Series(close)
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    volume_s = pd.Series(volume)
    
    # Donchian Channel (20-period)
    donchian_high = high_s.rolling(window=20, min_periods=20).max()
    donchian_low = low_s.rolling(window=20, min_periods=20).min()
    
    # Volume confirmation (20-period average)
    vol_ma = volume_s.rolling(window=20, min_periods=20).mean()
    
    # Weekly trend filter: EMA34 on 1w
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_1w_up = close_1w > ema34_1w
    trend_1w_down = close_1w < ema34_1w
    
    # Align weekly trend to 4h
    trend_1w_up_aligned = align_htf_to_ltf(prices, df_1w, trend_1w_up.astype(float))
    trend_1w_down_aligned = align_htf_to_ltf(prices, df_1w, trend_1w_down.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(vol_ma[i]) or
            np.isnan(trend_1w_up_aligned[i]) or np.isnan(trend_1w_down_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        volume_confirm = vol_ratio > 1.5
        
        # Conditions for long entry
        breakout_up = close[i] > donchian_high[i]
        
        # Conditions for short entry
        breakdown_down = close[i] < donchian_low[i]
        
        if position == 0:
            # Enter long: breakout up + weekly uptrend + volume
            if breakout_up and trend_1w_up_aligned[i] > 0.5 and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Enter short: breakdown down + weekly downtrend + volume
            elif breakdown_down and trend_1w_down_aligned[i] > 0.5 and volume_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit when price breaks below Donchian low or trend changes
            if close[i] < donchian_low[i] or trend_1w_up_aligned[i] < 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit when price breaks above Donchian high or trend changes
            if close[i] > donchian_high[i] or trend_1w_down_aligned[i] < 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals