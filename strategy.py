#!/usr/bin/env python3
"""
6h_ADX_Alligator_1dTrend_Filter
Hypothesis: Use ADX (trend strength) and Williams Alligator (trend direction) on 6h timeframe
filtered by 1d EMA trend. ADX > 25 indicates trending market; Alligator jaws/teeth/lips
alignment determines direction (bullish when lips > teeth > jaws, bearish when lips < teeth < jaws).
Works in both bull and bear markets by only trading when trend is strong and aligned.
Target: 20-40 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_williams_alligator(high, low, jaw_period=13, teeth_period=8, lips_period=5):
    """Calculate Williams Alligator lines (SMMA of median price)"""
    median_price = (high + low) / 2
    
    # Smoothed Moving Average (SMMA) calculation
    def smma(series, period):
        result = np.full_like(series, np.nan, dtype=np.float64)
        if len(series) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(series[:period])
        # Subsequent values: SMMA = (prev_smma * (period-1) + current_price) / period
        for i in range(period, len(series)):
            result[i] = (result[i-1] * (period-1) + series[i]) / period
        return result
    
    jaw = smma(median_price, jaw_period)
    teeth = smma(median_price, teeth_period)
    lips = smma(median_price, lips_period)
    
    return jaw, teeth, lips

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)"""
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Directional Movement
    dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                       np.maximum(high - np.roll(high, 1), 0), 0)
    dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    def smooth_wilder(series, period):
        result = np.full_like(series, np.nan, dtype=np.float64)
        if len(series) < period:
            return result
        # First value is sum of first 'period' values
        result[period-1] = np.nansum(series[:period])
        # Subsequent values: smoothed = prev_smoothed - (prev_smoothed/period) + current
        for i in range(period, len(series)):
            result[i] = result[i-1] - (result[i-1] / period) + series[i]
        return result
    
    tr_smoothed = smooth_wilder(tr, period)
    dm_plus_smoothed = smooth_wilder(dm_plus, period)
    dm_minus_smoothed = smooth_wilder(dm_minus, period)
    
    # Directional Indicators
    plus_di = np.where(tr_smoothed != 0, (dm_plus_smoothed / tr_smoothed) * 100, 0)
    minus_di = np.where(tr_smoothed != 0, (dm_minus_smoothed / tr_smoothed) * 100, 0)
    
    # DX and ADX
    dx = np.where((plus_di + minus_di) != 0, 
                  np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100, 0)
    
    # ADX is smoothed DX
    adx = smooth_wilder(dx, period)
    
    return adx

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Williams Alligator and ADX on 6h data
    jaw, teeth, lips = calculate_williams_alligator(high, low)
    adx = calculate_adx(high, low, close)
    
    # Get 1d EMA for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    ema_50 = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for all indicators
    start_idx = max(50, 20)  # EMA50 and vol20
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(adx[i]) or np.isnan(ema_50_aligned[i]) or 
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        # Williams Alligator signals
        # Bullish: lips > teeth > jaws (green alignment)
        alligator_bullish = lips[i] > teeth[i] and teeth[i] > jaw[i]
        # Bearish: lips < teeth < jaws (red alignment)
        alligator_bearish = lips[i] < teeth[i] and teeth[i] < jaw[i]
        
        # ADX trend strength filter
        strong_trend = adx[i] > 25
        
        # 1d EMA trend filter
        uptrend_1d = close[i] > ema_50_aligned[i]
        downtrend_1d = close[i] < ema_50_aligned[i]
        
        vol_conf = volume_confirm[i]
        
        if position == 0:
            # Long: Alligator bullish + strong trend + 1d uptrend + volume
            if alligator_bullish and strong_trend and uptrend_1d and vol_conf:
                signals[i] = size
                position = 1
            # Short: Alligator bearish + strong trend + 1d downtrend + volume
            elif alligator_bearish and strong_trend and downtrend_1d and vol_conf:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long: Alligator turns bearish OR ADX weakens
            if not alligator_bullish or adx[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: Alligator turns bullish OR ADX weakens
            if not alligator_bearish or adx[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_ADX_Alligator_1dTrend_Filter"
timeframe = "6h"
leverage = 1.0