#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1w EMA34 trend filter and volume confirmation.
Long when Bear Power < 0 (bulls in control) AND 1w EMA34 rising AND volume > 2.0x 20-period MA.
Short when Bull Power > 0 (bears in control) AND 1w EMA34 falling AND volume > 2.0x 20-period MA.
Exit when Elder Power reverses sign or 1w EMA34 slope changes.
Uses 1w HTF for major trend filter to avoid counter-trend trades in 2022 bear market, volume spike for momentum confirmation.
Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
Elder Ray measures bull/bear power via EMA13, works in both bull and bear markets by following higher timeframe trend.
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
    
    # Calculate 6h EMA13 for Elder Ray
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate 6h Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # Calculate 1w EMA34 for trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate 6h volume MA (20-period) for spike filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(13, 34, 20)  # EMA13, EMA34, volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema_1w_val = ema_34_1w_aligned[i]
        vol_ma_val = vol_ma_20[i]
        bull_val = bull_power[i]
        bear_val = bear_power[i]
        
        # Calculate 1w EMA34 slope for trend direction
        if i >= start_idx + 1:
            ema_1w_prev = ema_34_1w_aligned[i-1]
            ema_1w_rising = ema_1w_val > ema_1w_prev
            ema_1w_falling = ema_1w_val < ema_1w_prev
        else:
            ema_1w_rising = False
            ema_1w_falling = False
        
        # Volume filter: 6h volume > 2.0x 20-period MA (higher threshold for fewer trades)
        vol_filter = volume[i] > 2.0 * vol_ma_val
        
        if position == 0:
            # Long: Bear Power < 0 (bulls in control) AND 1w EMA34 rising AND volume filter
            if bear_val < 0 and ema_1w_rising and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: Bull Power > 0 (bears in control) AND 1w EMA34 falling AND volume filter
            elif bull_val > 0 and ema_1w_falling and vol_filter:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: Bear Power >= 0 (bulls losing control) OR 1w EMA34 starts falling
                if bear_val >= 0 or (i >= start_idx + 1 and ema_1w_val < ema_34_1w_aligned[i-1]):
                    exit_signal = True
            elif position == -1:
                # Short exit: Bull Power <= 0 (bears losing control) OR 1w EMA34 starts rising
                if bull_val <= 0 or (i >= start_idx + 1 and ema_1w_val > ema_34_1w_aligned[i-1]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_ElderRay_BullBearPower_1wEMA34_Trend_VolumeSpike"
timeframe = "6h"
leverage = 1.0