#!/usr/bin/env python3
"""
6h_ADX_Plus_Momentum_With_Volume_Filter
Hypothesis: Use ADX to filter trending markets (ADX>20) and combine with momentum (ROC>0) and volume spike (>1.5x 20-period avg) for entries. Exit when ADX drops below 15 or momentum turns negative. Designed for 6h timeframe to capture multi-day trends while avoiding whipsaws in low-volatility periods. Targets 15-25 trades/year by requiring strong trend confirmation, momentum alignment, and volume expansion. Works in bull markets by following uptrends with rising momentum, and in bear markets by taking shorts when ADX confirms downtrend and momentum is negative.
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
    
    # Calculate ADX (14-period)
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]  # first period
        
        # Directional Movement
        up_move = high - np.roll(high, 1)
        down_move = np.roll(low, 1) - low
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Smoothed values
        def smooth_wilder(arr, period):
            result = np.full_like(arr, np.nan)
            if len(arr) < period:
                return result
            # First value is simple average
            result[period-1] = np.nansum(arr[:period]) / period
            # Subsequent values Wilder smoothing
            for i in range(period, len(arr)):
                result[i] = (result[i-1] * (period-1) + arr[i]) / period
            return result
        
        atr = smooth_wilder(tr, period)
        plus_di = 100 * smooth_wilder(plus_dm, period) / atr
        minus_di = 100 * smooth_wilder(minus_dm, period) / atr
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = smooth_wilder(dx, period)
        return adx, plus_di, minus_di
    
    adx, plus_di, minus_di = calculate_adx(high, low, close, 14)
    
    # Rate of Change (ROC) - 10 period
    roc = np.full_like(close, np.nan)
    for i in range(10, n):
        if close[i-10] != 0:
            roc[i] = ((close[i] - close[i-10]) / close[i-10]) * 100
    
    # Volume confirmation: current volume > 1.5 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)  # need ADX and ROC
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(adx[i]) or np.isnan(roc[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long entry: ADX > 20 (trending), plus DI > minus DI, ROC > 0, volume confirmation
            if (adx[i] > 20 and plus_di[i] > minus_di[i] and roc[i] > 0 and vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: ADX > 20 (trending), minus DI > plus DI, ROC < 0, volume confirmation
            elif (adx[i] > 20 and minus_di[i] > plus_di[i] and roc[i] < 0 and vol_confirm[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: ADX drops below 15 (trend weakening) or ROC turns negative
            if adx[i] < 15 or roc[i] < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: ADX drops below 15 (trend weakening) or ROC turns positive
            if adx[i] < 15 or roc[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ADX_Plus_Momentum_With_Volume_Filter"
timeframe = "6h"
leverage = 1.0