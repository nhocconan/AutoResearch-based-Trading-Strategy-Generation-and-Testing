#!/usr/bin/env python3
"""
6h_ElderRay_With_Market_Regime
Hypothesis: Elder Ray Index (Bull Power/Bear Power) identifies bullish/bearish momentum.
Combine with market regime (ADX for trend strength, Bollinger Bands width for volatility).
In strong trends (ADX>25), trade Elder Ray direction. In range (ADX<20), fade extremes.
Designed for low trade frequency (<30/year) to avoid fee drag. Works in bull/bear via regime filter.
"""

name = "6h_ElderRay_With_Market_Regime"
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
    
    # EMA for Elder Ray (13-period)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power and Bear Power
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # ADX for trend strength (14-period)
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    plus_dm = np.insert(plus_dm, 0, 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    minus_dm = np.insert(minus_dm, 0, 0)
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    def wilders_smoothing(arr, period):
        result = np.full_like(arr, np.nan, dtype=np.float64)
        if len(arr) < period:
            return result
        result[period-1] = np.nansum(arr[:period])
        for i in range(period, len(arr)):
            result[i] = result[i-1] - (result[i-1] / period) + arr[i]
        return result
    
    period = 14
    plus_dm_smooth = wilders_smoothing(plus_dm, period)
    minus_dm_smooth = wilders_smoothing(minus_dm, period)
    tr_smooth = wilders_smoothing(tr, period)
    
    plus_di = 100 * plus_dm_smooth / tr_smooth
    minus_di = 100 * minus_dm_smooth / tr_smooth
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = wilders_smoothing(dx, period)
    
    # Bollinger Bands width for volatility regime (20-period)
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    bb_upper = sma_20 + 2 * std_20
    bb_lower = sma_20 - 2 * std_20
    bb_width = (bb_upper - bb_lower) / sma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Regime classification
        trending = adx[i] > 25
        ranging = adx[i] < 20
        
        if position == 0:
            # LONG conditions
            if trending and bull_power[i] > 0 and bull_power[i] > bull_power[i-1]:
                signals[i] = 0.25
                position = 1
            elif ranging and bear_power[i] < 0 and low[i] < bb_lower[i]:
                # Fade bear power exhaustion in range
                signals[i] = 0.25
                position = 1
            # SHORT conditions
            elif trending and bear_power[i] < 0 and bear_power[i] < bear_power[i-1]:
                signals[i] = -0.25
                position = -1
            elif ranging and bull_power[i] > 0 and high[i] > bb_upper[i]:
                # Fade bull power exhaustion in range
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG
            if trending and bull_power[i] <= 0:
                signals[i] = 0.0
                position = 0
            elif ranging and high[i] >= bb_upper[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT
            if trending and bear_power[i] >= 0:
                signals[i] = 0.0
                position = 0
            elif ranging and low[i] <= bb_lower[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals