#!/usr/bin/env python3
"""
6h_ElderRay_HTFTrend_VolumeSpike_V1
Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d EMA200 trend filter and volume spike confirmation (>2.0x 20-period volume MA). Long when Bull Power > 0 and Bear Power < 0 (bullish momentum) in uptrend; Short when Bull Power < 0 and Bear Power > 0 (bearish momentum) in downtrend. Uses discrete position sizing (0.25) to limit drawdown. Designed for low trade frequency (target: 50-150 total trades over 4 years) to minimize fee drag and work in both bull/bear markets via trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for EMA200 trend filter)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    # === 1d EMA200 for trend filter ===
    close_1d = df_1d['close'].values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # === 6h Indicators (primary timeframe) ===
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    volume_6h = df_6h['volume'].values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema_13 = pd.Series(close_6h).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high_6h - ema_13
    bear_power = low_6h - ema_13
    
    # Volume MA (20-period) for spike detection
    vol_ma = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):
        # Skip if indicators not ready
        if (np.isnan(ema_200_1d_aligned[i]) or np.isnan(bull_power[i]) 
            or np.isnan(bear_power[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_ok = volume_6h[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long: Bull Power > 0 AND Bear Power < 0 (bullish momentum) + uptrend + volume
            if bull_power[i] > 0 and bear_power[i] < 0 and close_6h[i] > ema_200_1d_aligned[i] and vol_ok:
                signals[i] = 0.25
                position = 1
            # Short: Bull Power < 0 AND Bear Power > 0 (bearish momentum) + downtrend + volume
            elif bull_power[i] < 0 and bear_power[i] > 0 and close_6h[i] < ema_200_1d_aligned[i] and vol_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: momentum reversal or loss of volume
            if bull_power[i] <= 0 or bear_power[i] >= 0 or not vol_ok:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: momentum reversal or loss of volume
            if bull_power[i] >= 0 or bear_power[i] <= 0 or not vol_ok:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_HTFTrend_VolumeSpike_V1"
timeframe = "6h"
leverage = 1.0