#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray Bull/Bear Power with 1d EMA50 trend filter and volume confirmation.
Elder Ray measures bull/bear power relative to EMA13. Long when bull power > 0 and rising, short when bear power < 0 and falling.
1d EMA50 ensures we trade with higher timeframe trend. Volume spike confirms momentum.
Designed for 6h timeframe to capture swing moves with moderate trade frequency.
Target: 50-150 total trades over 4 years (12-37/year).
Uses discrete position sizing (0.25) to balance return and fee drag.
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
    
    # Calculate EMA13 for Elder Ray (primary timeframe)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate volume spike: current volume > 1.8x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 1.8 * vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 13)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(ema_13[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: price relative to 1d EMA50
        trend_up = close[i] > ema_50_1d_aligned[i]
        trend_down = close[i] < ema_50_1d_aligned[i]
        
        if position == 0:
            # Long: Bull Power > 0 and rising (bull_power[i] > bull_power[i-1]), uptrend, volume spike
            if (bull_power[i] > 0 and bull_power[i] > bull_power[i-1] and 
                trend_up and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 and falling (bear_power[i] < bear_power[i-1]), downtrend, volume spike
            elif (bear_power[i] < 0 and bear_power[i] < bear_power[i-1] and 
                  trend_down and volume_spike[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Elder Ray power crosses zero or trend fails
            exit_signal = False
            if position == 1:
                # Exit long if bull power <= 0 or trend turns down
                if bull_power[i] <= 0 or not trend_up:
                    exit_signal = True
            elif position == -1:
                # Exit short if bear power >= 0 or trend turns up
                if bear_power[i] >= 0 or not trend_down:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_ElderRay_BullBearPower_1dEMA50_Trend_VolumeSpike"
timeframe = "6h"
leverage = 1.0