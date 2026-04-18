#!/usr/bin/env python3
"""
4h_12h_Camarilla_Pivot_R1S1_Breakout_Trend
Hypothesis: Trade breakouts above/below 12h Camarilla R1/S1 levels in direction of 4h EMA(34) trend, confirmed by volume >1.5x 20-period average. Uses 4h EMA trend filter to align with medium-term momentum. Position size 0.25 targeting ~30 trades/year to minimize fee drag. Works in bull/bear by trading breakouts with trend alignment and volume confirmation.
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
    
    # Get 12h data for Camarilla calculation
    df_12h = get_htf_data(prices, '12h')
    
    # Get 4h data for EMA trend filter
    df_4h = get_htf_data(prices, '4h')
    
    # 12h calculations (previous bar's OHLC)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Previous 12h bar's OHLC (completed bar)
    prev_high_12h = np.roll(high_12h, 1)
    prev_low_12h = np.roll(low_12h, 1)
    prev_close_12h = np.roll(close_12h, 1)
    prev_high_12h[0] = high_12h[0]
    prev_low_12h[0] = low_12h[0]
    prev_close_12h[0] = close_12h[0]
    
    # 12h Camarilla levels (based on previous bar)
    R1 = np.full_like(high_12h, np.nan)
    S1 = np.full_like(low_12h, np.nan)
    
    for i in range(1, len(high_12h)):
        range_ = prev_high_12h[i] - prev_low_12h[i]
        R1[i] = prev_close_12h[i] + range_ * 1.1 / 12
        S1[i] = prev_close_12h[i] - range_ * 1.1 / 12
    
    # 4h EMA trend filter (34-period)
    close_4h = df_4h['close'].values
    ema_period = 34
    ema_4h = np.full_like(close_4h, np.nan)
    
    if len(close_4h) >= ema_period:
        ema_4h[ema_period - 1] = np.mean(close_4h[:ema_period])
        for i in range(ema_period, len(close_4h)):
            ema_4h[i] = (close_4h[i] * 2 / (ema_period + 1)) + (ema_4h[i-1] * (ema_period - 1) / (ema_period + 1))
    
    # Align 12h Camarilla levels to 4h timeframe
    R1_aligned = align_htf_to_ltf(prices, df_12h, R1)
    S1_aligned = align_htf_to_ltf(prices, df_12h, S1)
    
    # Align 4h EMA to 4h timeframe (no alignment needed, but use for consistency)
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = np.full_like(volume, np.nan)
    vol_period = 20
    
    if len(volume) >= vol_period:
        for i in range(vol_period, len(volume)):
            vol_ma[i] = np.mean(volume[i - vol_period:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, vol_period)
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or 
            np.isnan(ema_4h_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: price breaks above R1 with volume and above 4h EMA
            if close[i] > R1_aligned[i] and vol_confirm and close[i] > ema_4h_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume and below 4h EMA
            elif close[i] < S1_aligned[i] and vol_confirm and close[i] < ema_4h_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price closes below S1 (reverse signal) or below 4h EMA
            if close[i] < S1_aligned[i] or close[i] < ema_4h_aligned[i]:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price closes above R1 (reverse signal) or above 4h EMA
            if close[i] > R1_aligned[i] or close[i] > ema_4h_aligned[i]:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_12h_Camarilla_Pivot_R1S1_Breakout_Trend"
timeframe = "4h"
leverage = 1.0