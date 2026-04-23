#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray Bull/Bear Power with 1w EMA50 trend filter and volume confirmation.
Long when Bull Power > 0 AND Bear Power < 0 (bullish momentum) AND price > 1w EMA50 AND volume > 1.3x 20-period average.
Short when Bull Power < 0 AND Bear Power > 0 (bearish momentum) AND price < 1w EMA50 AND volume > 1.3x 20-period average.
Exit when power convergence (|Bull Power| + |Bear Power| < 0.1 * ATR) or opposite power dominance.
Uses 1w HTF EMA for major trend alignment (avoids counter-trend trades). Target: 50-150 total trades over 4 years (12-37/year).
Elder Ray measures bull/bear power via EMA13; works in both bull/bear markets when aligned with higher timeframe trend.
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
    
    # Calculate 1w EMA50 for trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Elder Ray components (Bull Power, Bear Power) using EMA13
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13  # Bull Power = High - EMA13
    bear_power = low - ema13   # Bear Power = Low - EMA13
    
    # Calculate ATR(14) for convergence exit
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 20-period volume average for spike filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 13, 14)  # volume MA(20), EMA13, ATR(14)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(atr[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        ema_50 = ema_50_1w_aligned[i]
        bull = bull_power[i]
        bear = bear_power[i]
        atr_val = atr[i]
        vol_ma_val = vol_ma[i]
        
        if position == 0:
            # Long: Bull Power > 0 AND Bear Power < 0 (bullish) AND price > 1w EMA50 AND volume spike
            if bull > 0 and bear < 0 and price > ema_50 and volume[i] > 1.3 * vol_ma_val:
                signals[i] = 0.25
                position = 1
            # Short: Bull Power < 0 AND Bear Power > 0 (bearish) AND price < 1w EMA50 AND volume spike
            elif bull < 0 and bear > 0 and price < ema_50 and volume[i] > 1.3 * vol_ma_val:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: power convergence OR opposite power dominance
            power_sum = abs(bull) + abs(bear)
            convergence = power_sum < 0.1 * atr_val
            
            if position == 1:
                # Exit long if bearish dominance OR convergence
                if bull < 0 and bear > 0 or convergence:
                    exit_signal = True
                else:
                    exit_signal = False
            else:  # position == -1
                # Exit short if bullish dominance OR convergence
                if bull > 0 and bear < 0 or convergence:
                    exit_signal = True
                else:
                    exit_signal = False
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_ElderRay_BullBearPower_1wEMA50_Trend_VolumeConfirmation_PowerConvergenceExit"
timeframe = "6h"
leverage = 1.0