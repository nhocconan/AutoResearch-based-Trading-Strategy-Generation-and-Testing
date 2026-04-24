#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla H3/L3 breakout with 1d Supertrend (ATR=10, mult=3.0) trend filter, volume spike (>1.8x 20-bar average), and ATR regime filter (current ATR > 0.6x 50-bar average).
- Uses discrete position size 0.25 to limit drawdown and reduce fee churn.
- Targets 20-30 trades/year (80-120 total over 4 years) to stay fee-efficient.
- Supertrend adapts to volatility and trend, working in bull/bear markets.
- Volume and volatility filters avoid low-conviction entries.
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
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Prior 1d OHLC (completed 1d bar)
    high_1d = df_1d['high'].shift(1).values
    low_1d = df_1d['low'].shift(1).values
    close_1d = df_1d['close'].shift(1).values
    
    # Align to 4h timeframe
    high_1d_aligned = align_htf_to_ltf(prices, df_1d, high_1d)
    low_1d_aligned = align_htf_to_ltf(prices, df_1d, low_1d)
    close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
    
    # Calculate Camarilla levels
    camarilla_h3 = close_1d_aligned + 1.1 * (high_1d_aligned - low_1d_aligned) / 4
    camarilla_l3 = close_1d_aligned - 1.1 * (high_1d_aligned - low_1d_aligned) / 4
    
    # 1d Supertrend (ATR=10, mult=3.0) trend filter
    atr_period = 10
    multiplier = 3.0
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = 0
    atr = pd.Series(tr).ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
    
    # Basic Upper and Lower Bands
    hl2 = (high_1d + low_1d) / 2
    upper_band = hl2 + (multiplier * atr)
    lower_band = hl2 - (multiplier * atr)
    
    # Initialize Supertrend
    supertrend = np.zeros_like(close_1d)
    direction = np.ones_like(close_1d)  # 1 for uptrend, -1 for downtrend
    
    supertrend[0] = upper_band[0]
    direction[0] = 1
    
    for i in range(1, len(close_1d)):
        if close_1d[i] > supertrend[i-1]:
            direction[i] = 1
        elif close_1d[i] < supertrend[i-1]:
            direction[i] = -1
        else:
            direction[i] = direction[i-1]
        
        if direction[i] == 1 and direction[i-1] == -1:
            supertrend[i] = lower_band[i]
        elif direction[i] == -1 and direction[i-1] == 1:
            supertrend[i] = upper_band[i]
        elif direction[i] == 1:
            supertrend[i] = max(lower_band[i], supertrend[i-1])
        else:
            supertrend[i] = min(upper_band[i], supertrend[i-1])
    
    # Align Supertrend direction to 4h timeframe
    direction_aligned = align_htf_to_ltf(prices, df_1d, direction)
    
    # Volume confirmation: > 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR(14) for volatility regime filter
    atr_period_ltf = 14
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = 0
    atr_ltf = pd.Series(tr).rolling(window=atr_period_ltf, min_periods=atr_period_ltf).mean().values
    
    # ATR ratio: current ATR / 50-period average (avoid low volatility chop)
    atr_ma_long = pd.Series(atr_ltf).rolling(window=50, min_periods=50).mean().values
    atr_ratio = atr_ltf / np.where(atr_ma_long > 0, atr_ma_long, 1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20, atr_period_ltf, 50)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_h3[i]) or np.isnan(camarilla_l3[i]) or 
            np.isnan(direction_aligned[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(atr_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 1.8x average) + ATR ratio > 0.6 (avoid low vol)
        volume_confirm = volume[i] > 1.8 * vol_ma[i]
        vol_regime = atr_ratio[i] > 0.6
        
        if position == 0:
            # Long: Close > H3 AND Supertrend uptrend AND volume confirmation AND vol regime
            if close[i] > camarilla_h3[i] and direction_aligned[i] == 1 and volume_confirm and vol_regime:
                signals[i] = 0.25
                position = 1
            # Short: Close < L3 AND Supertrend downtrend AND volume confirmation AND vol regime
            elif close[i] < camarilla_l3[i] and direction_aligned[i] == -1 and volume_confirm and vol_regime:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Close < L3 OR Supertrend turns down
            if close[i] < camarilla_l3[i] or direction_aligned[i] == -1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Close > H3 OR Supertrend turns up
            if close[i] > camarilla_h3[i] or direction_aligned[i] == 1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_H3L3_Breakout_1dSupertrend10_3_VolumeATR_Filter_v1"
timeframe = "4h"
leverage = 1.0