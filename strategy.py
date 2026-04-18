#!/usr/bin/env python3
"""
4h_12h_Camarilla_Pivot_Breakout_Volume_Regime_v1
Hypothesis: Use 12h Camarilla pivot levels (R1/S1) for breakout direction, 12h EMA34 for trend filter, and volume confirmation for entry timing.
Go long when price breaks above 12h R1 AND 12h EMA34 is rising AND volume > 2x 20-period average.
Go short when price breaks below 12h S1 AND 12h EMA34 is falling AND volume > 2x 20-period average.
Exit on opposite breakout or when volume drops below average.
Uses 12h timeframe to reduce trade frequency and avoid overtrading.
Target: 20-40 trades/year by requiring multiple confluence factors.
Works in bull markets via breakout longs and in bear via breakdown shorts.
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
    
    # Get 12h data for Camarilla pivots and EMA
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h Camarilla pivot levels (based on previous day's range)
    # Camarilla: R1 = close + 1.1*(high-low)/12, S1 = close - 1.1*(high-low)/12
    # But we need to use the previous 12h bar's range
    prev_high_12h = np.roll(high_12h, 1)
    prev_low_12h = np.roll(low_12h, 1)
    prev_close_12h = np.roll(close_12h, 1)
    prev_high_12h[0] = np.nan
    prev_low_12h[0] = np.nan
    prev_close_12h[0] = np.nan
    
    # Calculate Camarilla levels
    R1_12h = prev_close_12h + 1.1 * (prev_high_12h - prev_low_12h) / 12
    S1_12h = prev_close_12h - 1.1 * (prev_high_12h - prev_low_12h) / 12
    
    # Calculate 12h EMA34 for trend filter
    ema_period = 34
    ema_12h = np.full_like(close_12h, np.nan)
    if len(close_12h) >= ema_period:
        ema_12h[ema_period-1] = np.mean(close_12h[:ema_period])
        for i in range(ema_period, len(close_12h)):
            ema_12h[i] = (close_12h[i] * 2 / (ema_period + 1)) + (ema_12h[i-1] * (ema_period - 1) / (ema_period + 1))
    
    # Calculate EMA slope for trend direction
    ema_slope_12h = np.diff(ema_12h, prepend=np.nan)
    
    # Align 12h indicators to 4h timeframe
    R1_12h_aligned = align_htf_to_ltf(prices, df_12h, R1_12h)
    S1_12h_aligned = align_htf_to_ltf(prices, df_12h, S1_12h)
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    ema_slope_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_slope_12h)
    
    # Volume confirmation: volume > 2x 20-period average
    vol_ma = np.full_like(volume, np.nan)
    vol_period = 20
    
    if len(volume) >= vol_period:
        for i in range(vol_period, len(volume)):
            vol_ma[i] = np.mean(volume[i - vol_period:i])
    
    vol_confirm = volume > 2 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(1, vol_period)  # Simple start index
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(R1_12h_aligned[i]) or np.isnan(S1_12h_aligned[i]) or 
            np.isnan(ema_12h_aligned[i]) or np.isnan(ema_slope_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R1 AND EMA rising AND volume confirmation
            if close[i] > R1_12h_aligned[i] and ema_slope_12h_aligned[i] > 0 and vol_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 AND EMA falling AND volume confirmation
            elif close[i] < S1_12h_aligned[i] and ema_slope_12h_aligned[i] < 0 and vol_confirm[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below S1 OR EMA turns down OR volume drops
            if close[i] < S1_12h_aligned[i] or ema_slope_12h_aligned[i] < 0 or not vol_confirm[i]:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above R1 OR EMA turns up OR volume drops
            if close[i] > R1_12h_aligned[i] or ema_slope_12h_aligned[i] > 0 or not vol_confirm[i]:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_12h_Camarilla_Pivot_Breakout_Volume_Regime_v1"
timeframe = "4h"
leverage = 1.0