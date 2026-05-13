#!/usr/bin/env python3
# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d trend filter (EMA50) and volume confirmation.
# Long when Bull Power > 0 (close > EMA13) AND Bear Power < 0 (close < EMA13) AND price > 1d EMA50 AND 1d volume > 1.5 * 20-period average volume.
# Short when Bear Power < 0 (close < EMA13) AND Bull Power < 0 (close < EMA13) AND price < 1d EMA50 AND 1d volume > 1.5 * 20-period average volume.
# Exit when Bull Power and Bear Power converge (|Bull Power - Bear Power| < 0.1 * ATR(14)) indicating weakening momentum.
# Uses discrete position sizing (0.25) to limit fee churn. Designed for BTC/ETH robustness by measuring bull/bear power relative to EMA13 with 1d trend and volume confirmation.
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.

name = "6h_ElderRay_TrendFilter_VolumeConfirm_v2"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1d EMA50 for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 1d volume spike filter (HTF)
    if len(df_1d) < 20:
        return np.zeros(n)
    volume_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume_1d > (1.5 * vol_ma_20)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike.astype(float))
    
    # Calculate Elder Ray components on primary timeframe
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).values
    atr14 = pd.Series(high - low).rolling(window=14, min_periods=14).mean().values
    bull_power = high - ema13  # Bull Power: High - EMA13
    bear_power = low - ema13   # Bear Power: Low - EMA13
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(14, n):  # Start after warmup period
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(volume_spike_aligned[i]) or
            np.isnan(bull_power[i]) or
            np.isnan(bear_power[i]) or
            np.isnan(atr14[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Bull Power > 0 AND Bear Power < 0 AND price > 1d EMA50 AND volume spike
            if (bull_power[i] > 0 and 
                bear_power[i] < 0 and 
                close[i] > ema_50_1d_aligned[i] and 
                volume_spike_aligned[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # SHORT: Bear Power < 0 AND Bull Power < 0 AND price < 1d EMA50 AND volume spike
            elif (bear_power[i] < 0 and 
                  bull_power[i] < 0 and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume_spike_aligned[i] > 0.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Bull Power and Bear Power converge (weakening momentum)
            if abs(bull_power[i] - bear_power[i]) < 0.1 * atr14[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Bull Power and Bear Power converge (weakening momentum)
            if abs(bull_power[i] - bear_power[i]) < 0.1 * atr14[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals