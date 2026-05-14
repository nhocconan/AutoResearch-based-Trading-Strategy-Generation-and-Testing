#!/usr/bin/env python3
# Hypothesis: 6h Elder Ray (Bull/Bear Power) + 12h EMA50 trend filter + 1d volume spike confirmation.
# Elder Ray measures bull/bear power relative to EMA13: Bull Power = High - EMA13, Bear Power = Low - EMA13.
# Long when Bull Power > 0 AND Bear Power rising (less negative) AND price > 12h EMA50 AND 1d volume > 2.0 * 20-period average volume.
# Short when Bear Power < 0 AND Bull Power falling (less positive) AND price < 12h EMA50 AND 1d volume > 2.0 * 20-period average volume.
# Exit when Elder Ray power crosses zero (Bull Power <= 0 for long, Bear Power >= 0 for short).
# Uses discrete position sizing (0.25) to limit fee churn. Target: 50-150 total trades over 4 years (12-37/year) for 6h.

name = "6h_ElderRay_BullBearPower_12hEMA50_Trend_1dVolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    open_ = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 12h EMA50 for trend filter (HTF)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate 1d volume confirmation filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    volume_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_confirm_1d = volume_1d > (2.0 * vol_ma_20_1d)
    volume_confirm_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_confirm_1d.astype(float))
    
    # Calculate Elder Ray (Bull/Bear Power) on 6h timeframe
    # Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # Calculate Elder Ray power derivatives (rate of change)
    # Bull Power rising = current > previous
    # Bear Power falling = current < previous (more negative)
    bull_power_rising = np.zeros(n, dtype=bool)
    bear_power_falling = np.zeros(n, dtype=bool)
    bull_power_rising[1:] = bull_power[1:] > bull_power[:-1]
    bear_power_falling[1:] = bear_power[1:] < bear_power[:-1]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(volume_confirm_1d_aligned[i]) or
            np.isnan(bull_power[i]) or
            np.isnan(bear_power[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Bull Power > 0 AND Bull Power rising AND price > 12h EMA50 AND volume confirmation
            if (bull_power[i] > 0 and 
                bull_power_rising[i] and 
                close[i] > ema_50_12h_aligned[i] and 
                volume_confirm_1d_aligned[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # SHORT: Bear Power < 0 AND Bear Power falling AND price < 12h EMA50 AND volume confirmation
            elif (bear_power[i] < 0 and 
                  bear_power_falling[i] and 
                  close[i] < ema_50_12h_aligned[i] and 
                  volume_confirm_1d_aligned[i] > 0.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Bull Power <= 0 (power dissipated)
            if bull_power[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Bear Power >= 0 (power dissipated)
            if bear_power[i] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals