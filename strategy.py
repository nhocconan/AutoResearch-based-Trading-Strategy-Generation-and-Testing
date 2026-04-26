#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_12hEMA50_Trend_VolumeSpike_v2
Hypothesis: On 4h timeframe, price breaking above/below Camarilla R1/S1 levels with volume spike and 12h EMA50 trend filter captures strong momentum moves. This combines proven price channel structure (Camarilla) with volume confirmation and HTF trend alignment for robustness in both bull and bear markets. Target: 75-200 total trades over 4 years (19-50/year).
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
    
    # Load 1d data ONCE before loop for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 12h EMA50 for HTF trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate volume average for spike detection (20-period)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need sufficient data for all indicators)
    start_idx = max(20, 50)  # need 20 for volume MA, 50 for EMA
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(volume_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Calculate Camarilla levels using previous 1d bar (completed bar)
        # Need to ensure we're using completed 1d bar - align_htf_to_ltf handles this
        if i < len(df_1d) * 16:  # rough check for sufficient 1d data
            idx_1d = min(i // 16, len(df_1d) - 1)
            if idx_1d > 0:
                prev_high = high_1d[idx_1d - 1]
                prev_low = low_1d[idx_1d - 1]
                prev_close = close_1d[idx_1d - 1]
                
                # Camarilla levels
                range_val = prev_high - prev_low
                if range_val > 0:
                    R1 = prev_close + (range_val * 1.1 / 12)
                    S1 = prev_close - (range_val * 1.1 / 12)
                    
                    # Volume confirmation
                    volume_spike = volume[i] > (volume_ma[i] * 1.5)
                    
                    # 12h trend filter
                    uptrend_12h = close[i] > ema_50_12h_aligned[i]
                    downtrend_12h = close[i] < ema_50_12h_aligned[i]
                    
                    # Long: price breaks above R1 with volume spike and uptrend
                    if close[i] > R1 and volume_spike and uptrend_12h:
                        if position != 1:
                            signals[i] = 0.25
                            position = 1
                        else:
                            signals[i] = 0.25
                    # Short: price breaks below S1 with volume spike and downtrend
                    elif close[i] < S1 and volume_spike and downtrend_12h:
                        if position != -1:
                            signals[i] = -0.25
                            position = -1
                        else:
                            signals[i] = -0.25
                    # Exit: loss of trend or price returns to midline
                    elif position == 1 and (close[i] < prev_close or not uptrend_12h):
                        signals[i] = 0.0
                        position = 0
                    elif position == -1 and (close[i] > prev_close or not downtrend_12h):
                        signals[i] = 0.0
                        position = 0
                    else:
                        # Hold current position
                        if position == 0:
                            signals[i] = 0.0
                        elif position == 1:
                            signals[i] = 0.25
                        else:
                            signals[i] = -0.25
                else:
                    # Hold current position if range invalid
                    if position == 0:
                        signals[i] = 0.0
                    elif position == 1:
                        signals[i] = 0.25
                    else:
                        signals[i] = -0.25
            else:
                # Hold current position if insufficient 1d data
                if position == 0:
                    signals[i] = 0.0
                elif position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
        else:
            # Hold current position if insufficient 1d data
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_12hEMA50_Trend_VolumeSpike_v2"
timeframe = "4h"
leverage = 1.0