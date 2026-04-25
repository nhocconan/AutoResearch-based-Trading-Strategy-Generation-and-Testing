#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike
Hypothesis: Camarilla R1/S1 breakout on 4h with 1d EMA50 trend filter and volume spike confirmation.
Only trade breakouts in direction of daily trend with above-average volume to avoid false breakouts in chop.
Uses discrete position sizing (0.25) for low trade frequency (~20-30/year) to minimize fee drag.
Works in both bull and bear markets by aligning with higher timeframe trend and requiring volume confirmation.
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
    
    # Get 1d data for HTF trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate EMA50 on 1d close for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF EMA50 to 4h timeframe (standard 1-bar delay for EMA)
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d, additional_delay_bars=1)
    
    # Calculate 20-period volume average for volume spike detection
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Camarilla levels from previous 1d bar
    # Camarilla: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    # where C, H, L are from previous completed 1d bar
    prev_close_1d = np.roll(close_1d, 1)
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d[0] = np.nan  # first bar has no previous
    prev_high_1d[0] = np.nan
    prev_low_1d[0] = np.nan
    
    camarilla_range = (prev_high_1d - prev_low_1d) * 1.1 / 12
    r1_1d = prev_close_1d + camarilla_range
    s1_1d = prev_close_1d - camarilla_range
    
    # Align Camarilla levels to 4h timeframe (standard 1-bar delay for HTF)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_1d, additional_delay_bars=1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_1d, additional_delay_bars=1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA50 (50) and volume MA (20)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50_aligned[i]) or 
            np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume spike condition: current volume > 1.5 * 20-period average
        volume_spike = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Look for Camarilla breakout signals with trend filter and volume confirmation
            # Long: price breaks above R1 in uptrend (close > EMA50) with volume spike
            # Short: price breaks below S1 in downtrend (close < EMA50) with volume spike
            long_signal = (close[i] > r1_aligned[i]) and (close[i] > ema50_aligned[i]) and volume_spike
            short_signal = (close[i] < s1_aligned[i]) and (close[i] < ema50_aligned[i]) and volume_spike
            
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
            # Exit when price moves back below EMA50 (trend reversal)
            exit_signal = close[i] < ema50_aligned[i]
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit when price moves back above EMA50 (trend reversal)
            exit_signal = close[i] > ema50_aligned[i]
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike"
timeframe = "4h"
leverage = 1.0