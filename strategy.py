#!/usr/bin/env python3
"""
1d_Williams_Alligator_ElderRay_1wTrend
Hypothesis: Williams Alligator (13,8,5 SMAs with 8,5,3 offsets) identifies trend direction and weakness. Elder Ray (bull/bear power via EMA13) measures trend strength. Combined with 1-week EMA50 trend filter and volume confirmation (1.5x average), this captures strong trending moves while avoiding false signals. Alligator provides trend/jam signals, Elder Ray filters weak moves, weekly trend ensures alignment with higher timeframe. Works in bull/bear by following weekly trend direction. Target: 20-50 trades/year per symbol.
"""

name = "1d_Williams_Alligator_ElderRay_1wTrend"
timeframe = "1d"
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
    
    # Get 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    # Williams Alligator (13,8,5 SMAs with 8,5,3 offsets)
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().shift(8)  # 13-period SMA shifted 8
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().shift(5)   # 8-period SMA shifted 5
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().shift(3)    # 5-period SMA shifted 3
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = EMA13 - Low
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = ema13 - low
    
    # 1-week EMA50 trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume spike: >1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after EMA50 warmup
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Lips > Teeth > Jaw (bullish alignment) + Bull Power > 0 + Weekly Uptrend + Volume Spike
            if (lips[i] > teeth[i] and teeth[i] > jaw[i] and 
                bull_power[i] > 0 and 
                close[i] > ema_50_1w_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Jaws > Teeth > Lips (bearish alignment) + Bear Power > 0 + Weekly Downtrend + Volume Spike
            elif (jaw[i] > teeth[i] and teeth[i] > lips[i] and 
                  bear_power[i] > 0 and 
                  close[i] < ema_50_1w_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Lips < Jaw (alligator sleeping) OR Bear Power > Bull Power (weakening)
            if lips[i] < jaw[i] or bear_power[i] > bull_power[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Jaws < Lips (alligator sleeping) OR Bull Power > Bear Power (weakening)
            if jaw[i] < lips[i] or bull_power[i] > bear_power[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals