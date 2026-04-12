#!/usr/bin/env python3
"""
12h_1d_Volume_Pivot_Range_v1
Hypothesis: Combine 1d pivot levels with volume confirmation and range-bound conditions to trade mean reversion in 12H timeframe.
Works in bull: buy near support, sell near resistance in uptrend.
Works in bear: sell near resistance, buy near support in downtrend.
Uses 1d pivot points (PP, R1, S1) for structure, volume spike for conviction, and RSI for overbought/oversold.
Targets 20-30 trades per year to minimize fee drag. Effective in ranging and trending markets with mean-reverting tendencies.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_Volume_Pivot_Range_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily data for pivots and volume average
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate daily pivot points: PP, R1, S1
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    pp = (daily_high + daily_low + daily_close) / 3
    r1 = 2 * pp - daily_low
    s1 = 2 * pp - daily_high
    
    # Align pivot levels to 12h
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Average daily volume for volume spike detection
    daily_volume = df_1d['volume'].values
    avg_vol = np.full(len(daily_volume), np.nan)
    for i in range(len(daily_volume)):
        if i >= 19:
            avg_vol[i] = np.mean(daily_volume[i-19:i+1])
    avg_vol_aligned = align_htf_to_ltf(prices, df_1d, avg_vol)
    
    # RSI(14) on 12h close for overbought/oversold
    def rsi(close, length=14):
        if len(close) < length + 1:
            return np.full(len(close), np.nan)
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.zeros_like(close)
        avg_loss = np.zeros_like(close)
        avg_gain[length] = np.mean(gain[:length])
        avg_loss[length] = np.mean(loss[:length])
        for i in range(length + 1, len(close)):
            avg_gain[i] = (avg_gain[i-1] * (length-1) + gain[i-1]) / length
            avg_loss[i] = (avg_loss[i-1] * (length-1) + loss[i-1]) / length
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
        rsi_vals = np.zeros_like(close)
        rsi_vals[:] = 100 - (100 / (1 + rs))
        rsi_vals[:length] = np.nan
        return rsi_vals
    
    rsi_vals = rsi(close, 14)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any data invalid
        if (np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(avg_vol_aligned[i]) or np.isnan(rsi_vals[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume spike: current volume > 1.5x average daily volume (scaled to 12h)
        # Approximate 12h volume as 1/2 of daily volume since 2x12h = 1d
        vol_spike = volume[i] > 1.5 * (avg_vol_aligned[i] / 2)
        
        # Price near pivot levels: within 0.5% of S1 (long) or R1 (short)
        near_s1 = abs(close[i] - s1_aligned[i]) / close[i] < 0.005
        near_r1 = abs(close[i] - r1_aligned[i]) / close[i] < 0.005
        
        # RSI conditions: oversold (<30) for long, overbought (>70) for short
        rsi_oversold = rsi_vals[i] < 30
        rsi_overbought = rsi_vals[i] > 70
        
        # Entry logic: mean reversion at pivot levels with volume confirmation
        long_entry = vol_spike and near_s1 and rsi_oversold
        short_entry = vol_spike and near_r1 and rsi_overbought
        
        # Exit logic: price moves to opposite pivot or RSI normalizes
        long_exit = close[i] >= pp_aligned[i] or rsi_vals[i] >= 50
        short_exit = close[i] <= pp_aligned[i] or rsi_vals[i] <= 50
        
        # Signal logic
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals