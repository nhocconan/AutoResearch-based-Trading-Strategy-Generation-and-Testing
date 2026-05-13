#!/usr/bin/env python3
# Hypothesis: 6h Elder Ray Index (Bull/Bear Power) with 1d trend filter (EMA50) and volume spike confirmation.
# Long when Bull Power > 0 AND Bear Power < 0 (bullish momentum) AND price > 1d EMA50 AND 1d volume > 2.0 * 20-period average volume.
# Short when Bear Power < 0 AND Bull Power > 0 (bearish momentum) AND price < 1d EMA50 AND 1d volume > 2.0 * 20-period average volume.
# Exit when Bull Power and Bear Power converge (|Bull Power - Bear Power| < 0.1 * ATR(14)) indicating loss of momentum.
# Uses discrete position sizing (0.25) to limit fee churn. Designed for BTC/ETH robustness by measuring bull/bear power relative to EMA13 with volume and trend confirmation.
# Target: 60-100 total trades over 4 years (15-25/year) for 6h timeframe.

name = "6h_ElderRay_TrendFilter_VolumeConfirm_v3"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate EMA13 for Elder Ray (primary timeframe)
    close_s = pd.Series(close)
    ema13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Calculate 1d EMA50 for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 1d volume spike filter (HTF)
    volume_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume_1d > (2.0 * vol_ma_20)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike.astype(float))
    
    # Calculate ATR(14) for exit condition (primary timeframe)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0  # First bar has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(14, n):  # Start after ATR/Elder Ray warmup
        # Skip if any required data is NaN
        if (np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or
            np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(volume_spike_aligned[i]) or
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Bull Power > 0 AND Bear Power < 0 (bullish) AND price > 1d EMA50 AND volume spike
            if (bull_power[i] > 0 and 
                bear_power[i] < 0 and 
                close[i] > ema_50_1d_aligned[i] and 
                volume_spike_aligned[i] > 0.5):  # True if volume spike aligned
                signals[i] = 0.25
                position = 1
            # SHORT: Bear Power < 0 AND Bull Power > 0 (bearish) AND price < 1d EMA50 AND volume spike
            elif (bear_power[i] < 0 and 
                  bull_power[i] > 0 and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume_spike_aligned[i] > 0.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Bull and Bear power converge (loss of momentum)
            power_diff = abs(bull_power[i] - bear_power[i])
            if power_diff < (0.1 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Bull and Bear power converge (loss of momentum)
            power_diff = abs(bull_power[i] - bear_power[i])
            if power_diff < (0.1 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals