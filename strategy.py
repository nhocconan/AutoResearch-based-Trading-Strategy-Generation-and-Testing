#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot reversal with weekly trend filter and volume confirmation.
# In ranging markets, price reverses from Camarilla H3/L3 levels.
# Uses weekly EMA50 for trend direction (counter-trend to weekly trend) and volume spike for confirmation.
# Works in bull markets (short at H3 in uptrend, long at L3 in downtrend) and bear markets.
# Target: 20-40 trades/year to avoid fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla multipliers
    H3 = close_1d + (high_1d - low_1d) * 1.1 / 6
    L3 = close_1d - (high_1d - low_1d) * 1.1 / 6
    H4 = close_1d + (high_1d - low_1d) * 1.1 / 2
    L4 = close_1d - (high_1d - low_1d) * 1.1 / 2
    
    # Align Camarilla levels to 6h timeframe
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    H4_aligned = align_htf_to_ltf(prices, df_1d, H4)
    L4_aligned = align_htf_to_ltf(prices, df_1d, L4)
    
    # Get weekly data for trend filter (EMA50)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Calculate EMA(50) on weekly close
    ema_50_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 50:
        ema_50_1w[49] = np.mean(close_1w[:50])
        alpha = 2 / (50 + 1)
        for i in range(50, len(close_1w)):
            ema_50_1w[i] = close_1w[i] * alpha + ema_50_1w[i-1] * (1 - alpha)
    
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup
    start_idx = max(20, 50)
    
    for i in range(start_idx, n):
        if (np.isnan(H3_aligned[i]) or 
            np.isnan(L3_aligned[i]) or
            np.isnan(H4_aligned[i]) or
            np.isnan(L4_aligned[i]) or
            np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Determine weekly trend direction
        # Use previous bar's EMA to avoid look-ahead
        if i > 0 and not np.isnan(ema_50_1w_aligned[i-1]):
            weekly_uptrend = ema_50_1w_aligned[i] > ema_50_1w_aligned[i-1]
            weekly_downtrend = ema_50_1w_aligned[i] < ema_50_1w_aligned[i-1]
        else:
            weekly_uptrend = False
            weekly_downtrend = False
        
        if position == 0:
            # Long entry: price at L3 with weekly downtrend (counter-trend) + volume spike
            if (close[i] <= L3_aligned[i] * 1.002 and  # Allow small buffer
                weekly_downtrend and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price at H3 with weekly uptrend (counter-trend) + volume spike
            elif (close[i] >= H3_aligned[i] * 0.998 and  # Allow small buffer
                  weekly_uptrend and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price reaches L4 (take profit) or weekly trend turns up
            if (close[i] >= L4_aligned[i] or 
                not weekly_downtrend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price reaches H4 (take profit) or weekly trend turns down
            if (close[i] <= H4_aligned[i] or 
                not weekly_uptrend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_H3L3_Reverse_WeeklyEMA50_Volume_v1"
timeframe = "6h"
leverage = 1.0