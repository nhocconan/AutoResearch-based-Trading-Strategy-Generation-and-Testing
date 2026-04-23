#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray Bull/Bear Power with 12h EMA50 trend filter and volume confirmation.
Long when Bull Power > 0 AND Bear Power < 0 AND close > 12h EMA50 AND volume > 1.5x 20-period average.
Short when Bear Power < 0 AND Bull Power > 0 AND close < 12h EMA50 AND volume > 1.5x 20-period average.
Exit when Elder Power signals reverse OR price closes below/above 12h EMA50.
Uses 12h HTF for trend alignment and 6h for Elder Ray calculation.
Target: ~12-25 trades/year on 6h timeframe with discrete sizing 0.25.
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
    
    # Calculate 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:  # Need enough for EMA
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate 13-period EMA for Elder Ray (13 is standard)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # 6h volume average (20-period) for spike filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 13, 50)  # vol_ma20, ema_13, ema_50_12h
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        ema_val = ema_50_12h_aligned[i]
        bull_val = bull_power[i]
        bear_val = bear_power[i]
        vol_ma_val = vol_ma[i]
        
        if position == 0:
            # Long: Bull Power > 0 AND Bear Power < 0 AND price > 12h EMA50 AND volume spike
            if bull_val > 0 and bear_val < 0 and price > ema_val and volume[i] > 1.5 * vol_ma_val:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 AND Bull Power > 0 AND price < 12h EMA50 AND volume spike
            elif bear_val < 0 and bull_val > 0 and price < ema_val and volume[i] > 1.5 * vol_ma_val:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            # Primary exit: Elder Power signals reverse
            if position == 1 and (bull_val <= 0 or bear_val >= 0):
                exit_signal = True
            elif position == -1 and (bull_val >= 0 or bear_val <= 0):
                exit_signal = True
            
            # Secondary exit: price closes below/above 12h EMA50 (trend change)
            if position == 1 and price < ema_val:
                exit_signal = True
            elif position == -1 and price > ema_val:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_ElderRay_BullBearPower_12hEMA50_Trend_VolumeConfirmation"
timeframe = "6h"
leverage = 1.0