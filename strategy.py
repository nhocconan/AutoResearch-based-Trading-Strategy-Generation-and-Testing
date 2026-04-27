#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla pivot (R1/S1) breakout with volume spike and 12h EMA trend filter.
- Camarilla levels provide precise support/resistance with institutional relevance
- Price breaking above R1 or below S1 signals potential breakout
- Volume spike confirms institutional participation
- 12h EMA50 filters for medium-term trend alignment
- Exit on opposite Camarilla level touch to avoid overtrading
- Target: 25-40 trades/year to minimize fee drag
"""

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
    
    # Get daily data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels (based on previous day)
    camarilla_high = np.full(len(high_1d), np.nan)
    camarilla_low = np.full(len(low_1d), np.nan)
    camarilla_range = np.full(len(high_1d), np.nan)
    
    for i in range(1, len(high_1d)):
        # Previous day's OHLC
        ph = high_1d[i-1]
        pl = low_1d[i-1]
        pc = close_1d[i-1]
        
        # Camarilla calculations
        range_val = ph - pl
        camarilla_high[i] = pc + (range_val * 1.1 / 12)  # R1
        camarilla_low[i] = pc - (range_val * 1.1 / 12)   # S1
        camarilla_range[i] = range_val
    
    camarilla_high_aligned = align_htf_to_ltf(prices, df_1d, camarilla_high)
    camarilla_low_aligned = align_htf_to_ltf(prices, df_1d, camarilla_low)
    camarilla_range_aligned = align_htf_to_ltf(prices, df_1d, camarilla_range)
    
    # Get 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_12h_50 = np.full(len(close_12h), np.nan)
    
    for i in range(len(close_12h)):
        if i >= 49:  # 50-period EMA
            if i == 49:
                ema_12h_50[i] = np.mean(close_12h[:50])
            else:
                ema_12h_50[i] = (close_12h[i] * 0.0377) + (ema_12h_50[i-1] * 0.9623)
        else:
            ema_12h_50[i] = np.nan
    
    ema_12h_50_aligned = align_htf_to_ltf(prices, df_12h, ema_12h_50)
    
    # Volume spike: current volume > 1.8 * 30-period average
    vol_ma_30 = np.full(n, np.nan)
    for i in range(30, n):
        vol_ma_30[i] = np.mean(volume[i-30:i])
    volume_spike = volume > (1.8 * vol_ma_30)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need enough data for all indicators
    start_idx = max(50, 60)
    
    for i in range(start_idx, n):
        if (np.isnan(camarilla_high_aligned[i]) or 
            np.isnan(camarilla_low_aligned[i]) or
            np.isnan(ema_12h_50_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long entry: price breaks above R1 + volume spike + price above 12h EMA50
            if (close[i] > camarilla_high_aligned[i] and 
                volume_spike[i] and 
                close[i] > ema_12h_50_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below S1 + volume spike + price below 12h EMA50
            elif (close[i] < camarilla_low_aligned[i] and 
                  volume_spike[i] and 
                  close[i] < ema_12h_50_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price touches or goes below S1 (opposite level)
            if close[i] <= camarilla_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price touches or goes above R1 (opposite level)
            if close[i] >= camarilla_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1S1_VolumeSpike_12hEMA50_Trend_v1"
timeframe = "4h"
leverage = 1.0