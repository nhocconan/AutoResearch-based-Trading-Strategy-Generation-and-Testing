#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray Index (Bull/Bear Power) with 1d trend filter and volume confirmation.
Long when Bull Power > 0 AND Bear Power < 0 AND 1d EMA50 rising AND volume > 1.5x 20-period MA.
Short when Bear Power < 0 AND Bull Power < 0 AND 1d EMA50 falling AND volume > 1.5x 20-period MA.
Exit when Elder Ray signals reverse or 1d EMA50 flips.
Uses 1d HTF for major trend alignment to avoid counter-trend trades, volume confirmation for momentum.
Elder Ray measures bull/bear power via EMA13, effective in both trending and ranging markets.
Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
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
    
    # Calculate EMA13 for Elder Ray
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Calculate 1d EMA50 for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume MA (20-period) for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(13, 50, 20)  # EMA13, EMA50, volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Elder Ray conditions
        bull_above_zero = bull_power[i] > 0
        bear_below_zero = bear_power[i] < 0
        
        # 1d EMA50 slope for trend direction
        if i >= start_idx + 1:
            ema_prev = ema_50_aligned[i-1]
            ema_rising = ema_50_aligned[i] > ema_prev
            ema_falling = ema_50_aligned[i] < ema_prev
        else:
            ema_rising = False
            ema_falling = False
        
        # Volume filter: > 1.5x 20-period MA
        vol_filter = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Long: Bull Power > 0 AND Bear Power < 0 AND 1d EMA50 rising AND volume filter
            if bull_above_zero and bear_below_zero and ema_rising and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 AND Bull Power < 0 AND 1d EMA50 falling AND volume filter
            elif bear_power[i] < 0 and bull_power[i] < 0 and ema_falling and vol_filter:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: Bull Power <= 0 OR Bear Power >= 0 OR 1d EMA50 starts falling
                if bull_power[i] <= 0 or bear_power[i] >= 0 or (i >= start_idx + 1 and ema_50_aligned[i] < ema_50_aligned[i-1]):
                    exit_signal = True
            elif position == -1:
                # Short exit: Bear Power >= 0 OR Bull Power >= 0 OR 1d EMA50 starts rising
                if bear_power[i] >= 0 or bull_power[i] >= 0 or (i >= start_idx + 1 and ema_50_aligned[i] > ema_50_aligned[i-1]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_ElderRay_BullBearPower_1dEMA50_Trend_VolumeConfirmation"
timeframe = "6h"
leverage = 1.0