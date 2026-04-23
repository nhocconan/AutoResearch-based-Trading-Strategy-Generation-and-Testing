#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1w trend filter and volume confirmation.
Long when Bull Power > 0 AND Bear Power < 0 (bullish momentum) AND 1w EMA34 rising AND volume > 2x 20-period MA.
Short when Bear Power < 0 AND Bull Power < 0 (bearish momentum) AND 1w EMA34 falling AND volume > 2x 20-period MA.
Exit when Elder Power signals reverse or volume drops.
Uses 1w EMA34 for major trend filter to avoid counter-trend trades, Elder Ray for momentum, volume spike for confirmation.
Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
Elder Ray captures bull/bear power via EMA13, effective in both trending and ranging markets.
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
    
    # Bull Power = High - EMA13, Bear Power = Low - EMA13
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
        
        # Calculate 1w EMA34 slope for trend direction (rising/falling)
        if i >= start_idx + 1:
            ema_prev = ema_34_1w_aligned[i-1]
            ema_rising = ema_34_1w_aligned[i] > ema_prev
            ema_falling = ema_34_1w_aligned[i] < ema_prev
        else:
            ema_rising = False
            ema_falling = False
        
        # Volume filter: 6h volume > 2x 20-period MA (adaptive to volatility)
        vol_filter = volume[i] > 2.0 * vol_ma_20[i]
        
        if position == 0:
            # Long: Bull Power > 0 AND Bear Power < 0 (bullish momentum) AND 1w EMA34 rising AND volume filter
            if bull_power[i] > 0 and bear_power[i] < 0 and ema_rising and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 AND Bull Power < 0 (bearish momentum) AND 1w EMA34 falling AND volume filter
            elif bear_power[i] < 0 and bull_power[i] < 0 and ema_falling and vol_filter:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: Bull Power <= 0 OR Bear Power >= 0 (momentum fades) OR 1w EMA34 starts falling
                if bull_power[i] <= 0 or bear_power[i] >= 0 or (i >= start_idx + 1 and ema_34_1w_aligned[i] < ema_34_1w_aligned[i-1]):
                    exit_signal = True
            elif position == -1:
                # Short exit: Bull Power >= 0 OR Bear Power >= 0 (momentum fades) OR 1w EMA34 starts rising
                if bull_power[i] >= 0 or bear_power[i] >= 0 or (i >= start_idx + 1 and ema_34_1w_aligned[i] > ema_34_1w_aligned[i-1]):
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