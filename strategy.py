#!/usr/bin/env python3
"""
6h_ThreeBarReversal_VolumeTrend
Hypothesis: Three-bar reversal patterns (bullish: 3 higher highs, bearish: 3 lower lows) with volume confirmation and trend filter capture momentum shifts. Works in bull/bear by only taking reversals in the direction of the 12h trend, filtering counter-trend noise.
"""

name = "6h_ThreeBarReversal_VolumeTrend"
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
    
    # Get 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    
    # 12h EMA50 trend filter
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume spike: >1.5x 20-period average (6h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after EMA50 warmup
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Bullish 3-bar reversal: 3 consecutive higher highs
            bullish_reversal = (high[i-2] < high[i-1]) and (high[i-1] < high[i])
            # Bearish 3-bar reversal: 3 consecutive lower lows
            bearish_reversal = (low[i-2] > low[i-1]) and (low[i-1] > low[i])
            
            # LONG: Bullish reversal + above 12h EMA50 + volume spike
            if bullish_reversal and (close[i] > ema_50_12h_aligned[i]) and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Bearish reversal + below 12h EMA50 + volume spike
            elif bearish_reversal and (close[i] < ema_50_12h_aligned[i]) and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Bearish reversal or price below 12h EMA50
            bearish_reversal = (low[i-2] > low[i-1]) and (low[i-1] > low[i])
            if bearish_reversal or (close[i] < ema_50_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Bullish reversal or price above 12h EMA50
            bullish_reversal = (high[i-2] < high[i-1]) and (high[i-1] < high[i])
            if bullish_reversal or (close[i] > ema_50_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals