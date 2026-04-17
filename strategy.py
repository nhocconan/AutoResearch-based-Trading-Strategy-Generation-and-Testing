#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d EMA50 trend filter and volume confirmation.
Long when Bull Power > 0, Bear Power < 0, price > 1d EMA50, and volume > 1.5x 6h avg volume.
Short when Bear Power < 0, Bull Power < 0, price < 1d EMA50, and volume > 1.5x 6h avg volume.
Exit when Elder Ray signals weaken (Bull Power <= 0 for long, Bear Power >= 0 for short).
Uses 6h for execution and volume, 1d for EMA trend and Elder Ray calculation.
Elder Ray measures bull/bear strength relative to EMA13, effective in both trending and ranging markets.
Target: 12-30 trades/year per symbol.
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
    
    # Get 1d data for EMA trend filter and Elder Ray calculation
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1d_rising = ema_50_1d > np.roll(ema_50_1d, 1)
    ema_50_1d_falling = ema_50_1d < np.roll(ema_50_1d, 1)
    ema_50_1d_rising[0] = False
    ema_50_1d_falling[0] = False
    
    # Calculate 1d EMA(13) for Elder Ray
    ema_13_1d = pd.Series(close_1d).ewm(span=13, min_periods=13, adjust=False).mean().values
    
    # Calculate Elder Ray components: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power_1d = high_1d - ema_13_1d
    bear_power_1d = low_1d - ema_13_1d
    
    # Get 6h data for execution and volume
    df_6h = get_htf_data(prices, '6h')
    volume_6h = df_6h['volume'].values
    
    # Calculate 6h volume MA (20-period)
    vol_ma_20 = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    
    # Align 1d indicators to 6h timeframe
    ema_50_1d_rising_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d_rising)
    ema_50_1d_falling_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d_falling)
    bull_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_6h, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100  # warmup period
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_50_1d_rising_aligned[i]) or 
            np.isnan(ema_50_1d_falling_aligned[i]) or
            np.isnan(bull_power_1d_aligned[i]) or
            np.isnan(bear_power_1d_aligned[i]) or
            np.isnan(vol_ma_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 6h volume > 1.5x 20-bar average
        volume_confirmed = volume[i] > 1.5 * vol_ma_20_aligned[i]
        
        if position == 0:
            # Long: Bull Power > 0, Bear Power < 0, price above rising 1d EMA50, volume confirmed
            if (bull_power_1d_aligned[i] > 0 and 
                bear_power_1d_aligned[i] < 0 and 
                close[i] > ema_50_1d_aligned[i] if 'ema_50_1d_aligned' in locals() else close[i] > ema_50_1d[i // 24] and  # fallback for aligned
                volume_confirmed and 
                ema_50_1d_rising_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0, Bull Power < 0, price below falling 1d EMA50, volume confirmed
            elif (bear_power_1d_aligned[i] < 0 and 
                  bull_power_1d_aligned[i] < 0 and 
                  close[i] < ema_50_1d_aligned[i] if 'ema_50_1d_aligned' in locals() else close[i] < ema_50_1d[i // 24] and  # fallback for aligned
                  volume_confirmed and 
                  ema_50_1d_falling_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Bull Power <= 0 (weakening bullish strength) or price below EMA50
            if bull_power_1d_aligned[i] <= 0 or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Bear Power >= 0 (weakening bearish strength) or price above EMA50
            if bear_power_1d_aligned[i] >= 0 or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_EMA50Trend_Volume_Confirm"
timeframe = "6h"
leverage = 1.0