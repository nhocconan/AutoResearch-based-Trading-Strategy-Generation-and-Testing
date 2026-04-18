#!/usr/bin/env python3
"""
4h_1d_Camarilla_Pivot_R1S1_Breakout_Volume
Hypothesis: Long when price breaks above Camarilla R1 level with volume >1.5x 20-period average, short when breaks below S1 with volume confirmation. Uses 1d EMA(50) to filter counter-trend trades: only long when price > 1d EMA(50), only short when price < 1d EMA(50). Position size 0.25, targeting 20-40 trades/year to avoid fee drag. Works in bull/bear by trading breakouts with trend alignment and volume confirmation.
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
    
    # Get 1d data for Camarilla pivot calculation and EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # 1d calculations
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous 1d bar's OHLC (completed bar)
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d = np.roll(close_1d, 1)
    prev_high_1d[0] = high_1d[0]
    prev_low_1d[0] = low_1d[0]
    prev_close_1d[0] = close_1d[0]
    
    # 1d Camarilla pivot levels (R1 and S1)
    # Pivot point = (H + L + C) / 3
    # R1 = Pivot + (H - L) * 1.1 / 12
    # S1 = Pivot - (H - L) * 1.1 / 12
    pp = (prev_high_1d + prev_low_1d + prev_close_1d) / 3
    r1 = pp + (prev_high_1d - prev_low_1d) * 1.1 / 12
    s1 = pp - (prev_high_1d - prev_low_1d) * 1.1 / 12
    
    # 1d EMA trend filter (50-period)
    ema_period = 50
    ema_1d = np.full_like(close_1d, np.nan)
    
    if len(close_1d) >= ema_period:
        # Calculate EMA using standard formula
        multiplier = 2 / (ema_period + 1)
        ema_1d[ema_period - 1] = np.mean(close_1d[:ema_period])
        for i in range(ema_period, len(close_1d)):
            ema_1d[i] = (close_1d[i] * multiplier) + (ema_1d[i-1] * (1 - multiplier))
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = np.full_like(volume, np.nan)
    vol_period = 20
    
    if len(volume) >= vol_period:
        for i in range(vol_period, len(volume)):
            vol_ma[i] = np.mean(volume[i - vol_period:i])
    
    # Align 1d levels to 4h timeframe (wait for 1d bar to close)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, ema_period, vol_period)
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: price breaks above R1 with volume and above 1d EMA
            if close[i] > r1_aligned[i] and vol_confirm and close[i] > ema_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume and below 1d EMA
            elif close[i] < s1_aligned[i] and vol_confirm and close[i] < ema_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price closes below S1 (reverse signal) or below 1d EMA
            if close[i] < s1_aligned[i] or close[i] < ema_1d_aligned[i]:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price closes above R1 (reverse signal) or above 1d EMA
            if close[i] > r1_aligned[i] or close[i] > ema_1d_aligned[i]:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_1d_Camarilla_Pivot_R1S1_Breakout_Volume"
timeframe = "4h"
leverage = 1.0