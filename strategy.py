#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_12hTrend_VolumeSpike
Hypothesis: Camarilla R1/S1 breakouts on 4h with 12h EMA50 trend filter and volume spike confirmation.
Only trade breakouts in direction of 12h trend. Uses discrete position sizing (0.30) to minimize fee churn.
Designed for low trade frequency (~20-40/year) to work in both bull and bear markets via trend alignment.
Camarilla levels provide high-probability reversal/breakout points, volume confirms institutional interest,
and 12h EMA50 filter ensures we trade with the intermediate-term trend.
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
    
    # Get 12h data for HTF trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate EMA50 on 12h close for trend filter
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h, additional_delay_bars=1)
    
    # Calculate 20-period volume average for volume spike detection
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Camarilla levels from previous day's OHLC (using 1d data)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Get previous day's high, low, close for Camarilla calculation
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla R1 and S1 levels
    # R1 = close + (high - low) * 1.12 / 12
    # S1 = close - (high - low) * 1.12 / 12
    camarilla_range = high_1d - low_1d
    r1_level = close_1d + (camarilla_range * 1.12 / 12)
    s1_level = close_1d - (camarilla_range * 1.12 / 12)
    
    # Align Camarilla levels to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_level, additional_delay_bars=1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_level, additional_delay_bars=1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA50 (50) and volume MA (20)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50_12h_aligned[i]) or 
            np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i]) or
            np.isnan(vol_ma20[i])):
            signals[i] = 0.0 if position == 0 else (0.30 if position == 1 else -0.30)
            continue
        
        # Volume spike condition: current volume > 1.5 * 20-period average
        volume_spike = volume[i] > 1.5 * vol_ma20[i]
        
        if position == 0:
            # Look for Camarilla breakout signals with trend filter and volume confirmation
            # Long: price breaks above R1 in uptrend (close > EMA50) with volume spike
            # Short: price breaks below S1 in downtrend (close < EMA50) with volume spike
            long_signal = (close[i] > r1_aligned[i]) and (close[i] > ema50_12h_aligned[i]) and volume_spike
            short_signal = (close[i] < s1_aligned[i]) and (close[i] < ema50_12h_aligned[i]) and volume_spike
            
            if long_signal:
                signals[i] = 0.30
                position = 1
            elif short_signal:
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.30
            # Exit when price moves back below EMA50 (trend reversal) or breaks below S1 (failed breakout)
            exit_signal = (close[i] < ema50_12h_aligned[i]) or (close[i] < s1_aligned[i])
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.30
            # Exit when price moves back above EMA50 (trend reversal) or breaks above R1 (failed breakout)
            exit_signal = (close[i] > ema50_12h_aligned[i]) or (close[i] > r1_aligned[i])
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_12hTrend_VolumeSpike"
timeframe = "4h"
leverage = 1.0