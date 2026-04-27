#!/usr/bin/env python3
"""
6h ADX + Williams Alligator Combination with 12h Trend Filter.
Long when ADX > 25, price above Alligator's Jaw (teeth), and 12h trend up.
Short when ADX > 25, price below Alligator's Jaw, and 12h trend down.
Exit when ADX < 20 (weak trend) or price crosses back below/above Jaw.
Uses Williams Alligator (SMMA: 13,8,5) as dynamic support/resistance.
Designed for low frequency (15-30 trades/year) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def smma(source, length):
    """Smoothed Moving Average (used in Williams Alligator)"""
    if length < 1:
        return source
    result = np.full_like(source, np.nan, dtype=np.float64)
    if len(source) < length:
        return result
    # First value is simple average
    result[length-1] = np.mean(source[:length])
    # Subsequent values: SMMA = (PREV_SMMA * (length-1) + CURRENT) / length
    for i in range(length, len(source)):
        result[i] = (result[i-1] * (length-1) + source[i]) / length
    return result

def calculate_adx(high, low, close, period=14):
    """Calculate Average Directional Index (ADX)"""
    if len(high) < period + 1:
        return np.full_like(high, np.nan, dtype=np.float64)
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Directional Movement
    dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                       np.maximum(high - np.roll(high, 1), 0), 0)
    dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    atr = np.zeros_like(high)
    dm_plus_smooth = np.zeros_like(high)
    dm_minus_smooth = np.zeros_like(high)
    
    # Initial averages
    atr[period-1] = np.mean(tr[:period])
    dm_plus_smooth[period-1] = np.mean(dm_plus[:period])
    dm_minus_smooth[period-1] = np.mean(dm_minus[:period])
    
    # Wilder's smoothing
    for i in range(period, len(high)):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        dm_plus_smooth[i] = (dm_plus_smooth[i-1] * (period-1) + dm_plus[i]) / period
        dm_minus_smooth[i] = (dm_minus_smooth[i-1] * (period-1) + dm_minus[i]) / period
    
    # Directional Indicators
    di_plus = np.zeros_like(high)
    di_minus = np.zeros_like(high)
    dx = np.zeros_like(high)
    
    # Avoid division by zero
    valid_atr = atr != 0
    di_plus[valid_atr] = (dm_plus_smooth[valid_atr] / atr[valid_atr]) * 100
    di_minus[valid_atr] = (dm_minus_smooth[valid_atr] / atr[valid_atr]) * 100
    
    # DX
    di_sum = di_plus + di_minus
    valid_di_sum = di_sum != 0
    dx[valid_di_sum] = (np.abs(di_plus[valid_di_sum] - di_minus[valid_di_sum]) / di_sum[valid_di_sum]) * 100
    
    # ADX (smoothed DX)
    adx = np.zeros_like(high)
    if len(dx) >= 2*period-1:
        adx[2*period-2] = np.mean(dx[period-1:2*period-1])
        for i in range(2*period-1, len(high)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
    
    return adx

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = np.full_like(close_12h, np.nan, dtype=np.float64)
    if len(close_12h) >= 50:
        ema_50_12h[49] = np.mean(close_12h[:50])
        for i in range(50, len(close_12h)):
            ema_50_12h[i] = (close_12h[i] * 2 + ema_50_12h[i-1] * 49) / 51
    
    # Align 12h EMA50 to 6h timeframe
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Williams Alligator components on 6h timeframe
    median_price = (high + low) / 2
    
    jaw_raw = smma(median_price, 13)
    teeth_raw = smma(median_price, 8)
    lips_raw = smma(median_price, 5)
    
    # Align Alligator components (no additional delay needed for SMMA)
    jaw_aligned = align_htf_to_ltf(prices, pd.DataFrame({'high': high, 'low': low, 'close': close}), jaw_raw)
    teeth_aligned = align_htf_to_ltf(prices, pd.DataFrame({'high': high, 'low': low, 'close': close}), teeth_raw)
    lips_aligned = align_htf_to_ltf(prices, pd.DataFrame({'high': high, 'low': low, 'close': close}), lips_raw)
    
    # Calculate ADX on 6h timeframe
    adx = calculate_adx(high, low, close, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need Alligator (13), ADX (2*14-1=27), and EMA50 (50)
    start_idx = max(13, 27, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(adx[i])):
            signals[i] = 0.0
            continue
        
        # Current price and indicators
        price_now = close[i]
        jaw = jaw_aligned[i]
        teeth = teeth_aligned[i]
        ema_50_12h = ema_50_12h_aligned[i]
        adx_value = adx[i]
        
        # Trend filter: 12h EMA50 direction
        trend_up = price_now > ema_50_12h
        trend_down = price_now < ema_50_12h
        
        # ADX filters: strong trend (>25) and weak trend (<20) for exit
        strong_trend = adx_value > 25
        weak_trend = adx_value < 20
        
        if position == 0:
            # Bull: ADX > 25, price above Jaw, price above EMA50_12h
            if strong_trend and price_now > jaw and trend_up:
                signals[i] = size
                position = 1
            # Bear: ADX > 25, price below Jaw, price below EMA50_12h
            elif strong_trend and price_now < jaw and trend_down:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: ADX < 20 (weak trend) or price crosses below Jaw
            if weak_trend or price_now < jaw:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: ADX < 20 (weak trend) or price crosses above Jaw
            if weak_trend or price_now > jaw:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_ADX_WilliamsAlligator_12hTrend"
timeframe = "6h"
leverage = 1.0