#!/usr/bin/env python3
"""
4h_Keltner_Breakout_TrendFilter_VolumeConfirm
Hypothesis: Price breaking above/below Keltner Channel (EMA20 + ATRx2) with EMA50 trend filter and volume confirmation (1.5x average) captures strong trending moves while avoiding false breakouts. Keltner adapts to volatility, EMA50 ensures trend alignment, and volume filter adds confirmation. Target: 20-40 trades/year per symbol. Works in bull/bear by following trend direction.
"""

name = "4h_Keltner_Breakout_TrendFilter_VolumeConfirm"
timeframe = "4h"
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
    
    # Keltner Channel: EMA20 ± ATR(10)*2
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_10 = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    kc_upper = ema_20 + (2 * atr_10)
    kc_lower = ema_20 - (2 * atr_10)
    
    # Volume spike: >1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # EMA50 trend filter (same timeframe)
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        if (np.isnan(kc_upper[i]) or np.isnan(kc_lower[i]) or 
            np.isnan(ema_50[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Keltner Upper + EMA50 uptrend + volume spike
            if (close[i] > kc_upper[i] and 
                close[i] > ema_50[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Keltner Lower + EMA50 downtrend + volume spike
            elif (close[i] < kc_lower[i] and 
                  close[i] < ema_50[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below EMA20 (middle of Keltner)
            if close[i] < ema_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above EMA20 (middle of Keltner)
            if close[i] > ema_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals