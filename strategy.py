#!/usr/bin/env python3
# 12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeS
# Hypothesis: Breakout of Camarilla R1/S1 levels with 1d trend filter and volume confirmation.
# Long when price breaks above R1 with 1d uptrend and volume > 1.5x average.
# Short when price breaks below S1 with 1d downtrend and volume > 1.5x average.
# Works in bull/bear by following daily trend and using volume to confirm institutional interest.
# Target: 15-30 trades/year per symbol.

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeS"
timeframe = "12h"
leverage = 1.0

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
    
    # 12h indicators
    # Pivot points from previous bar
    pivot = (high[:-1] + low[:-1] + close[:-1]) / 3
    range_ = high[:-1] - low[:-1]
    
    # Camarilla levels
    R1 = close[:-1] + range_ * 1.1 / 12
    S1 = close[:-1] - range_ * 1.1 / 12
    
    # Align levels to current bar (breakout uses previous bar's levels)
    R1 = np.concatenate([np.full(1, np.nan), R1])
    S1 = np.concatenate([np.full(1, np.nan), S1])
    
    # Volume average (20-period)
    volume_s = pd.Series(volume)
    vol_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Daily trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    daily_uptrend = close_1d > ema50_1d
    daily_downtrend = close_1d < ema50_1d
    
    # Align daily trend to 12h
    daily_uptrend_aligned = align_htf_to_ltf(prices, df_1d, daily_uptrend.astype(float))
    daily_downtrend_aligned = align_htf_to_ltf(prices, df_1d, daily_downtrend.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 25
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(R1[i]) or np.isnan(S1[i]) or np.isnan(vol_ma[i]) or
            np.isnan(daily_uptrend_aligned[i]) or np.isnan(daily_downtrend_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        volume_confirm = vol_ratio > 1.5
        
        daily_up = daily_uptrend_aligned[i] > 0.5
        daily_down = daily_downtrend_aligned[i] > 0.5
        
        if position == 0:
            # Enter long: daily uptrend + price breaks above R1 + volume confirmation
            if daily_up and volume_confirm and close[i] > R1[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: daily downtrend + price breaks below S1 + volume confirmation
            elif daily_down and volume_confirm and close[i] < S1[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price returns below R1 or trend changes
            if close[i] < R1[i] or not daily_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price returns above S1 or trend changes
            if close[i] > S1[i] or not daily_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals