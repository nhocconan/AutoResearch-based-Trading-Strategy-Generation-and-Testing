#!/usr/bin/env python3
"""
6h_ElderRay_BullBearPower_Regime
Hypothesis: Elder Ray Bull/Bear Power with regime filter (ADX > 20) captures
trend momentum in both bull and bear markets. Bull Power > 0 and Bear Power < 0
with price > EMA20 for longs, and vice versa for shorts. ADX ensures we only
trade in trending conditions, reducing whipsaw. Position size 0.25 targets
~20-40 trades/year to minimize fee drag.
"""

name = "6h_ElderRay_BullBearPower_Regime"
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
    
    # EMA20 for trend filter
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    # ADX for trend strength (14-period)
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            plus_dm[i] = max(0, high[i] - high[i-1])
            minus_dm[i] = max(0, low[i-1] - low[i])
            if plus_dm[i] == minus_dm[i]:
                plus_dm[i] = 0
                minus_dm[i] = 0
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Smooth with Wilder's smoothing (alpha = 1/period)
        atr = np.zeros_like(high)
        plus_di = np.zeros_like(high)
        minus_di = np.zeros_like(high)
        
        atr[period-1] = np.mean(tr[:period])
        plus_dm_smooth = np.zeros_like(high)
        minus_dm_smooth = np.zeros_like(high)
        plus_dm_smooth[period-1] = np.mean(plus_dm[:period])
        minus_dm_smooth[period-1] = np.mean(minus_dm[:period])
        
        for i in range(period, len(high)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
            plus_dm_smooth[i] = (plus_dm_smooth[i-1] * (period-1) + plus_dm[i]) / period
            minus_dm_smooth[i] = (minus_dm_smooth[i-1] * (period-1) + minus_dm[i]) / period
        
        plus_di = 100 * plus_dm_smooth / atr
        minus_di = 100 * minus_dm_smooth / atr
        dx = np.zeros_like(high)
        dx[period:] = 100 * np.abs(plus_di[period:] - minus_di[period:]) / (plus_di[period:] + minus_di[period:])
        adx = np.zeros_like(high)
        adx[2*period-1] = np.mean(dx[period:2*period])
        for i in range(2*period, len(high)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        return adx
    
    adx = calculate_adx(high, low, close, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after warmup
        if position == 0:
            # LONG: Bull Power > 0, price > EMA20, ADX > 20 (trending)
            if (bull_power[i] > 0 and 
                close[i] > ema20[i] and 
                adx[i] > 20):
                signals[i] = 0.25
                position = 1
            # SHORT: Bear Power < 0, price < EMA20, ADX > 20 (trending)
            elif (bear_power[i] < 0 and 
                  close[i] < ema20[i] and 
                  adx[i] > 20):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Bull Power <= 0 or price < EMA20 or ADX < 20 (no trend)
            if (bull_power[i] <= 0) or \
               (close[i] < ema20[i]) or \
               (adx[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Bear Power >= 0 or price > EMA20 or ADX < 20 (no trend)
            if (bear_power[i] >= 0) or \
               (close[i] > ema20[i]) or \
               (adx[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals