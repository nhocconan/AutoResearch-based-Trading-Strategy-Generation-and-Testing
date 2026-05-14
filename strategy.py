#!/usr/bin/env python3
# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 12h EMA trend filter and 1d volume confirmation.
# Long when Bull Power > 0 AND price > 12h EMA50 AND 1d volume > 1.5 * 20-period average volume.
# Short when Bear Power < 0 AND price < 12h EMA50 AND 1d volume > 1.5 * 20-period average volume.
# Exit when Elder Power reverses sign (Bull Power <= 0 for long exit, Bear Power >= 0 for short exit).
# Uses discrete position sizing (0.25) to balance return and drawdown. Designed for 6h timeframe to avoid fee drag while capturing institutional momentum with trend and volume confirmation.

name = "6h_ElderRay_TrendFilter_VolumeConfirm_v4"
timeframe = "6h"
leverage = 1.0

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
    
    # Calculate 12h EMA50 for trend filter (HTF)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    close_12h = df_12h['close'].values
    ema_50 = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50)
    
    # Calculate 1d volume spike filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    volume_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume_1d > (1.5 * vol_ma_20)  # Volume above 1.5x average
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike.astype(float))
    
    # Calculate Elder Ray (Bull Power and Bear Power) on 6h data
    # Bull Power = High - EMA13
    # Bear Power = Low - EMA13
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(13, n):  # Start after EMA13 warmup
        # Skip if any required data is NaN
        if (np.isnan(ema_50_aligned[i]) or 
            np.isnan(volume_spike_aligned[i]) or
            np.isnan(bull_power[i]) or
            np.isnan(bear_power[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Bull Power > 0 AND price > 12h EMA50 AND volume spike
            if (bull_power[i] > 0 and 
                close[i] > ema_50_aligned[i] and 
                volume_spike_aligned[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # SHORT: Bear Power < 0 AND price < 12h EMA50 AND volume spike
            elif (bear_power[i] < 0 and 
                  close[i] < ema_50_aligned[i] and 
                  volume_spike_aligned[i] > 0.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Bull Power <= 0 (momentum weakening)
            if bull_power[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Bear Power >= 0 (momentum weakening)
            if bear_power[i] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals