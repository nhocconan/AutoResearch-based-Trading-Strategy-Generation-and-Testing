#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla Pivot Breakout with Volume Confirmation and 1d Trend Filter.
# Price breaking above R1 or below S1 with volume spike and aligned with daily trend.
# Works in both bull (breakouts up) and bear (breakouts down) markets.
# Target: 20-30 trades/year to avoid fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for pivot points and EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate previous day's Camarilla pivot levels
    camarilla_r1 = np.full(len(df_1d), np.nan)
    camarilla_s1 = np.full(len(df_1d), np.nan)
    
    for i in range(1, len(df_1d)):
        # Use previous day's data to calculate today's pivot levels (no look-ahead)
        ph = high_1d[i-1]
        pl = low_1d[i-1]
        pc = close_1d[i-1]
        range_ = ph - pl
        
        camarilla_r1[i] = pc + (range_ * 1.1 / 12)  # R1 level
        camarilla_s1[i] = pc - (range_ * 1.1 / 12)  # S1 level
    
    # Align Camarilla levels to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Calculate EMA(50) on daily close for trend filter
    ema_50_1d = np.full(len(df_1d), np.nan)
    alpha = 2 / (50 + 1)
    for i in range(len(close_1d)):
        if i < 49:
            ema_50_1d[i] = np.mean(close_1d[:i+1]) if i > 0 else close_1d[i]
        else:
            if np.isnan(ema_50_1d[i-1]):
                ema_50_1d[i] = np.mean(close_1d[i-49:i+1])
            else:
                ema_50_1d[i] = close_1d[i] * alpha + ema_50_1d[i-1] * (1 - alpha)
    
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need enough data for indicators
    start_idx = max(20, 50)
    
    for i in range(start_idx, n):
        if (np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend direction from daily EMA50
        # Use previous bar's EMA to avoid look-ahead
        if i > 0 and not np.isnan(ema_50_1d_aligned[i-1]):
            trend_up = ema_50_1d_aligned[i] > ema_50_1d_aligned[i-1]
            trend_down = ema_50_1d_aligned[i] < ema_50_1d_aligned[i-1]
        else:
            trend_up = False
            trend_down = False
        
        if position == 0:
            # Long entry: price breaks above R1 + uptrend + volume spike
            if (close[i] > r1_aligned[i] and 
                trend_up and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below S1 + downtrend + volume spike
            elif (close[i] < s1_aligned[i] and 
                  trend_down and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price falls back below R1 or trend turns down
            if (close[i] < r1_aligned[i] or 
                not trend_up):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price rises back above S1 or trend turns up
            if (close[i] > s1_aligned[i] or 
                not trend_down):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_1dEMA50_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0