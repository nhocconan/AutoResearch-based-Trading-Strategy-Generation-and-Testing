#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_12hTrendFilter_VolumeSpike_v1
Hypothesis: Trade 4h Camarilla R1/S1 breakouts with 12h EMA50 trend filter and volume confirmation.
Only long when price breaks above R1 and 12h EMA50 is rising; only short when price breaks below S1 and 12h EMA50 is falling.
Volume must be > 1.5 * ATR to confirm momentum. Uses discrete sizing 0.25 to minimize fee drag.
Target: 20-50 trades/year on BTC/ETH/SOL. Works in bull via breakouts, in bear via short breakdowns.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate ATR for volume spike and Camarilla levels (using 4h data)
    tr1 = np.maximum(high[1:] - low[1:], np.abs(high[1:] - close[:-1]))
    tr2 = np.maximum(np.abs(low[1:] - close[:-1]), tr1)
    tr = np.concatenate([[np.inf], tr2])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate Camarilla levels from previous day (using 1d data)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Use previous 1d bar's high/low/close for Camarilla calculation
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    camarilla_R1 = close_1d + (high_1d - low_1d) * 1.1 / 12
    camarilla_S1 = close_1d - (high_1d - low_1d) * 1.1 / 12
    
    # Align 1d Camarilla levels to 4h timeframe (these levels are valid for the entire day)
    camarilla_R1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R1)
    camarilla_S1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0
    
    # Start index: need warmup for 12h EMA50 (50) and ATR (14)
    start_idx = max(50, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(camarilla_R1_aligned[i]) or np.isnan(camarilla_S1_aligned[i]) or
            np.isnan(atr[i])):
            signals[i] = 0.0
            bars_since_entry = 0
            continue
        
        # Volume spike: current volume > 1.5 * ATR
        volume_spike = volume[i] > 1.5 * atr[i]
        
        # 12h EMA50 trend: rising if current > previous, falling if current < previous
        if i > start_idx:
            ema_50_12h_prev = ema_50_12h_aligned[i-1]
            ema_rising = ema_50_12h_aligned[i] > ema_50_12h_prev
            ema_falling = ema_50_12h_aligned[i] < ema_50_12h_prev
        else:
            ema_rising = False
            ema_falling = False
        
        if position == 0:
            # Long setup: price breaks above R1 AND EMA50 rising AND volume spike
            long_setup = (close[i] > camarilla_R1_aligned[i]) and ema_rising and volume_spike
            
            # Short setup: price breaks below S1 AND EMA50 falling AND volume spike
            short_setup = (close[i] < camarilla_S1_aligned[i]) and ema_falling and volume_spike
            
            if long_setup:
                signals[i] = 0.25
                position = 1
                bars_since_entry = 0
            elif short_setup:
                signals[i] = -0.25
                position = -1
                bars_since_entry = 0
            else:
                signals[i] = 0.0
                bars_since_entry = 0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            bars_since_entry += 1
            # Exit: price breaks below S1 OR EMA50 turns falling OR max hold (12 bars = 3 days)
            if (close[i] < camarilla_S1_aligned[i]) or (not ema_rising) or (bars_since_entry >= 12):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            bars_since_entry += 1
            # Exit: price breaks above R1 OR EMA50 turns rising OR max hold (12 bars = 3 days)
            if (close[i] > camarilla_R1_aligned[i]) or (not ema_falling) or (bars_since_entry >= 12):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_12hTrendFilter_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0