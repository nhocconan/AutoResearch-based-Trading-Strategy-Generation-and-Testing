#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1w trend filter and volume confirmation.
Long when Bear Power > 0 (bulls in control) AND 1w EMA50 rising AND volume > 1.3x 20-period MA.
Short when Bull Power < 0 (bears in control) AND 1w EMA50 falling AND volume > 1.3x 20-period MA.
Exit when Elder Power reverses or 1w EMA50 changes slope.
Uses 1w HTF for major trend to avoid counter-trend trades, Elder Ray for momentum, volume for confirmation.
Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
Elder Ray shows who controls the market, 1w EMA50 filters major trend, volume confirms strength.
Works in both bull (follow trend) and bear (fade counter-trend spikes) markets.
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
    
    # Calculate 6h EMA13 for Elder Ray (EMA13 close)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # Calculate 1w EMA50 for trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 6h volume MA (20-period) for spike filter
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
        
        bp = bull_power[i]  # Bull Power
        br = bear_power[i]  # Bear Power
        ema_val = ema_50_aligned[i]
        vol_ma_val = vol_ma_20[i]
        
        # Calculate 1w EMA50 slope for trend direction (rising/falling)
        if i >= start_idx + 1:
            ema_prev = ema_50_aligned[i-1]
            ema_rising = ema_val > ema_prev
            ema_falling = ema_val < ema_prev
        else:
            ema_rising = False
            ema_falling = False
        
        # Volume filter: 6h volume > 1.3x 20-period MA (adaptive to volatility)
        vol_filter = volume[i] > 1.3 * vol_ma_val
        
        if position == 0:
            # Long: Bull Power > 0 (bulls in control) AND EMA50 rising AND volume filter
            if bp > 0 and ema_rising and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 (bears in control) AND EMA50 falling AND volume filter
            elif br < 0 and ema_falling and vol_filter:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: Bull Power <= 0 (bulls lose control) OR EMA50 starts falling
                if bp <= 0 or (i >= start_idx + 1 and ema_val < ema_50_aligned[i-1]):
                    exit_signal = True
            elif position == -1:
                # Short exit: Bear Power >= 0 (bears lose control) OR EMA50 starts rising
                if br >= 0 or (i >= start_idx + 1 and ema_val > ema_50_aligned[i-1]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_ElderRay_BullBearPower_1wEMA50_Trend_VolumeFilter"
timeframe = "6h"
leverage = 1.0