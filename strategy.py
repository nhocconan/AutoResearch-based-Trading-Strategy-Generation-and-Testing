#!/usr/bin/env python3
# 4H_Camarilla_R1_S1_Breakout_1dTrend_VolumeS
# Hypothesis: Trade Camarilla pivot reversals at R1/S1 with daily trend filter and volume confirmation.
# Long when: price closes below S1 during daily uptrend with volume > 1.5x average, targeting R1.
# Short when: price closes above R1 during daily downtrend with volume > 1.5x average, targeting S1.
# Uses daily EMA50 for trend filter and 4h Camarilla levels from prior day.
# Works in bull/bear by following daily trend and using volume to confirm institutional interest.
# Target: 20-40 trades/year per symbol.

name = "4H_Camarilla_R1_S1_Breakout_1dTrend_VolumeS"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prrices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla levels from previous day's data
    # We need daily OHLC to calculate Camarilla levels
    # Since we're on 4h timeframe, we'll calculate daily levels from 1d data
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels for each day
    # Camarilla formulas:
    # H = (high + low + close) / 3
    # R1 = H + 1.1 * (high - low) / 12
    # S1 = H - 1.1 * (high - low) / 12
    # We'll use previous day's OHLC for current day's levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot and ranges
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    R1_1d = pivot_1d + 1.1 * range_1d / 12.0
    S1_1d = pivot_1d - 1.1 * range_1d / 12.0
    
    # Align daily Camarilla levels to 4h
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1_1d)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1_1d)
    
    # Daily trend filter using EMA50
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    daily_uptrend = close_1d > ema50_1d
    daily_downtrend = close_1d < ema50_1d
    
    # Align daily trend to 4h
    daily_uptrend_aligned = align_htf_to_ltf(prices, df_1d, daily_uptrend.astype(float))
    daily_downtrend_aligned = align_htf_to_ltf(prices, df_1d, daily_downtrend.astype(float))
    
    # Volume confirmation (20-period average)
    volume_s = pd.Series(volume)
    vol_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or
            np.isnan(daily_uptrend_aligned[i]) or np.isnan(daily_downtrend_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        volume_confirm = vol_ratio > 1.5
        
        daily_up = daily_uptrend_aligned[i] > 0.5
        daily_down = daily_downtrend_aligned[i] > 0.5
        
        if position == 0:
            # Enter long: price closes below S1 during daily uptrend with volume confirmation
            if daily_up and volume_confirm:
                if close[i] <= S1_aligned[i]:
                    signals[i] = 0.25
                    position = 1
            # Enter short: price closes above R1 during daily downtrend with volume confirmation
            elif daily_down and volume_confirm:
                if close[i] >= R1_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Exit long: price reaches R1 or trend changes
            if close[i] >= R1_aligned[i] or not daily_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price reaches S1 or trend changes
            if close[i] <= S1_aligned[i] or not daily_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals