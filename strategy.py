#!/usr/bin/env python3
"""
1d_Camarilla_R1S1_Breakout_1wTrendFilter_VolumeSpike
Hypothesis: Trade daily Camarilla R1/S1 breakouts with weekly EMA50 trend filter and volume confirmation.
Designed for 1d timeframe to target 30-100 trades over 4 years (7-25/year) with discrete sizing (0.25) to minimize fee drag.
Uses weekly trend filter to avoid counter-trend trades in bear markets (2022 crash, 2025 range) and volume spike for momentum confirmation.
Exit on re-entry to Camarilla range or trend reversal. Prioritizes BTC/ETH edge with SOL as secondary.
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
    
    # Get weekly data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Get daily data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous daily bar
    typical_price_1d = (df_1d['high'].values + df_1d['low'].values + df_1d['close'].values) / 3
    range_1d = df_1d['high'].values - df_1d['low'].values
    
    # Camarilla R1 and S1 levels (main breakout levels)
    camarilla_r1 = typical_price_1d + (range_1d * 1.1 / 12)
    camarilla_s1 = typical_price_1d - (range_1d * 1.1 / 12)
    
    # Align Camarilla levels to 1d timeframe (use previous day's levels)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Calculate ATR for volume spike filter (adaptive to volatility)
    tr1 = np.maximum(high[1:] - low[1:], np.abs(high[1:] - close[:-1]))
    tr2 = np.maximum(np.abs(low[1:] - close[:-1]), tr1)
    tr = np.concatenate([[np.inf], tr2])  # first TR undefined
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for weekly EMA50 (50) and ATR (14)
    start_idx = max(50, 14, 1)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        # Volume spike: current volume > 1.5 * ATR (adaptive threshold)
        volume_spike = volume[i] > 1.5 * atr[i]
        
        if position == 0:
            # Long: price breaks above R1 AND weekly trend bullish (close > EMA50) AND volume spike
            long_setup = (close[i] > camarilla_r1_aligned[i]) and \
                         (close[i] > ema_50_1w_aligned[i]) and \
                         volume_spike
            # Short: price breaks below S1 AND weekly trend bearish (close < EMA50) AND volume spike
            short_setup = (close[i] < camarilla_s1_aligned[i]) and \
                          (close[i] < ema_50_1w_aligned[i]) and \
                          volume_spike
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: price re-enters between S1 and R1 OR weekly trend turns bearish
            if (camarilla_s1_aligned[i] < close[i] < camarilla_r1_aligned[i]) or \
               (close[i] < ema_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price re-enters between S1 and R1 OR weekly trend turns bullish
            if (camarilla_s1_aligned[i] < close[i] < camarilla_r1_aligned[i]) or \
               (close[i] > ema_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Camarilla_R1S1_Breakout_1wTrendFilter_VolumeSpike"
timeframe = "1d"
leverage = 1.0