#!/usr/bin/env python3
"""
1d_Camarilla_R1S1_Breakout_1wTrend_VolumeSpike
Hypothesis: Daily Camarilla R1/S1 breakout with weekly trend filter (price > weekly EMA50 for long, < weekly EMA50 for short) and volume confirmation (>2.0x 20-day mean volume). Uses HTF 1w for trend alignment to capture long-term momentum while reducing whipsaw. Volume confirmation ensures breakouts have conviction. Discrete position sizing (0.25) minimizes fee churn. Target: 7-25 trades/year per symbol. Effective in both bull (breakouts with volume) and bear (trend-following via shorts) markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for HTF trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate EMA(50) on weekly for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Align EMA50 to daily timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Camarilla levels from previous weekly bar (HLC of prior bar)
    camarilla_r1 = close_1w + 1.1 * (high_1w - low_1w)  # R1 = C + 1.1*(H-L)
    camarilla_s1 = close_1w - 1.1 * (high_1w - low_1w)  # S1 = C - 1.1*(H-L)
    
    # Align Camarilla levels to daily timeframe (use previous bar's levels)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s1)
    
    # Volume confirmation: current volume > 2.0x 20-bar mean volume
    vol_mean_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_mean_20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA and volume mean
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(vol_mean_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        if position == 0:
            # Long: price breaks above Camarilla R1 in uptrend (price > weekly EMA50) with volume confirmation
            # Short: price breaks below Camarilla S1 in downtrend (price < weekly EMA50) with volume confirmation
            long_signal = (close[i] > camarilla_r1_aligned[i]) and (close[i] > ema_50_aligned[i]) and vol_confirm[i]
            short_signal = (close[i] < camarilla_s1_aligned[i]) and (close[i] < ema_50_aligned[i]) and vol_confirm[i]
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit when price moves back below weekly EMA50 (trend reversal)
            exit_signal = close[i] < ema_50_aligned[i]
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit when price moves back above weekly EMA50 (trend reversal)
            exit_signal = close[i] > ema_50_aligned[i]
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Camarilla_R1S1_Breakout_1wTrend_VolumeSpike"
timeframe = "1d"
leverage = 1.0