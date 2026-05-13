#!/usr/bin/env python3
"""
6h_Elder_Ray_BullPower_Force_With_Volume_Regime
Hypothesis: Elder Ray Bull/Bear Power measures bullish/bearish momentum relative to EMA13. 
In trending markets (ADX > 25), strong Bull Power (> 0) with volume confirmation signals long; 
strong Bear Power (< 0) with volume confirmation signals short. 
Uses 6h timeframe with 13-period EMA for power calculation and 14-period ADX for trend filter.
Designed for low trade frequency (~15-30/year) to minimize fee dash in 6-hour bars.
"""

name = "6h_Elder_Ray_BullPower_Force_With_Volume_Regime"
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
    
    # Calculate EMA13 for Elder Ray (using close)
    close_series = pd.Series(close)
    ema13 = close_series.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema13  # Bull Power = High - EMA13
    bear_power = low - ema13   # Bear Power = Low - EMA13
    
    # ADX calculation for trend strength (using Wilder's smoothing)
    # Calculate True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0  # First value has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Calculate Directional Movement
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    up_move[0] = 0
    down_move[0] = 0
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smooth TR, +DM, -DM using Wilder's smoothing (alpha = 1/period)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        alpha = 1.0 / period
        # First value is simple average
        result[period-1] = np.nanmean(data[:period])
        for i in range(period, len(data)):
            result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
        return result
    
    tr14 = wilders_smoothing(tr, 14)
    plus_dm14 = wilders_smoothing(plus_dm, 14)
    minus_dm14 = wilders_smoothing(minus_dm, 14)
    
    # Calculate DI+ and DI-
    plus_di14 = 100 * plus_dm14 / tr14
    minus_di14 = 100 * minus_dm14 / tr14
    
    # Calculate DX and ADX
    dx = 100 * np.abs(plus_di14 - minus_di14) / (plus_di14 + minus_di14)
    adx = wilders_smoothing(dx, 14)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if ADX not available (NaN from smoothing)
        if np.isnan(adx[i]):
            signals[i] = 0.0
            continue
            
        if position == 0:
            # LONG: ADX > 25 (trending), Bull Power > 0 (bullish momentum), volume confirmation
            if (adx[i] > 25 and 
                bull_power[i] > 0 and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: ADX > 25 (trending), Bear Power < 0 (bearish momentum), volume confirmation
            elif (adx[i] > 25 and 
                  bear_power[i] < 0 and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: ADX < 20 (trend weakening) OR Bull Power <= 0 (lost bullish momentum)
            if (adx[i] < 20 or 
                bull_power[i] <= 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: ADX < 20 (trend weakening) OR Bear Power >= 0 (lost bearish momentum)
            if (adx[i] < 20 or 
                bear_power[i] >= 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals