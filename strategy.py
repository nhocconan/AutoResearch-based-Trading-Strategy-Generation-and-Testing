#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray Bull/Bear Power with 1d EMA50 trend filter and volume confirmation.
Uses 6h Elder Ray (Bull Power = High - EMA13, Bear Power = Low - EMA13) to measure
buying/selling pressure behind price moves. Combined with 1d EMA50 trend filter to
avoid counter-trend trades. Volume spike confirms momentum. Designed for 6h timeframe
to capture swing moves with moderate trade frequency. Works in both bull and bear
markets by adapting to the prevailing trend via EMA50 filter.
Target: 50-150 total trades over 4 years = 12-37/year.
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
    
    # Calculate 6h EMA13 for Elder Ray
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 13:
        return np.zeros(n)
    
    close_6h = df_6h['close'].values
    ema_13_6h = pd.Series(close_6h).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema_13_6h_aligned = align_htf_to_ltf(prices, df_6h, ema_13_6h)
    
    # Calculate 1d EMA50 for primary trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Elder Ray components: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema_13_6h_aligned
    bear_power = low - ema_13_6h_aligned
    
    # Calculate volume spike: current volume > 2.0x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # need EMA50 and volume MA20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_13_6h_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: 1d EMA50 direction
        trend_up = close[i] > ema_50_1d_aligned[i]
        trend_down = close[i] < ema_50_1d_aligned[i]
        
        if position == 0:
            # Long: Bull Power > 0 (buying pressure) AND uptrend AND volume spike
            if bull_power[i] > 0 and trend_up and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 (selling pressure) AND downtrend AND volume spike
            elif bear_power[i] < 0 and trend_down and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Elder Ray divergence - weakening momentum
            exit_signal = False
            if position == 1:
                # Exit long when Bull Power turns negative (buying pressure fading)
                if bull_power[i] <= 0:
                    exit_signal = True
            elif position == -1:
                # Exit short when Bear Power turns positive (selling pressure fading)
                if bear_power[i] >= 0:
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