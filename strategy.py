#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray Index with 12h EMA50 trend filter and volume spike confirmation.
- Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
- Bull Power > 0 AND Bear Power < 0 indicates balanced momentum (avoid chop)
- Strong Bull Power (> 0.5*ATR) with volume > 1.5x average signals bullish entry
- Strong Bear Power (< -0.5*ATR) with volume > 1.5x average signals bearish entry
- 12h EMA50 ensures trades align with higher timeframe trend
- Volume confirmation reduces false breakouts
- Discrete position size 0.25 to manage drawdown
- Target: 12-25 trades/year on 6h timeframe (50-100 total over 4 years)
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
    
    # EMA13 for Elder Ray calculation
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # ATR(14) for power threshold
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Elder Ray components
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 50)  # volume MA, 12h EMA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or np.isnan(atr_14[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(ema_50_12h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 1.5x average)
        volume_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Power thresholds
        strong_bull = bull_power[i] > 0.5 * atr_14[i]
        strong_bear = bear_power[i] < -0.5 * atr_14[i]
        balanced = bull_power[i] > 0 and bear_power[i] < 0
        
        if position == 0:
            # Long: Strong bull power, balanced market, above 12h EMA50, volume confirmation
            if strong_bull and balanced and close[i] > ema_50_12h_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Strong bear power, balanced market, below 12h EMA50, volume confirmation
            elif strong_bear and balanced and close[i] < ema_50_12h_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Weak bull power OR price crosses below 12h EMA50
            if bull_power[i] < 0 or close[i] < ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Weak bear power OR price crosses above 12h EMA50
            if bear_power[i] > 0 or close[i] > ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_Alligator_VolumeConfirm_v2"
timeframe = "6h"
leverage = 1.0