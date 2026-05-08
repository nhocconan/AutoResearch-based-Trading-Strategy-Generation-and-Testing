#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator with 1d ADX trend filter and volume confirmation.
# Long when green line > red line > blue line (bullish alignment) AND 1d ADX > 25 AND volume > 1.5x 20-period average.
# Short when green line < red line < blue line (bearish alignment) AND 1d ADX > 25 AND volume > 1.5x 20-period average.
# Exit when Alligator lines crossover (green crosses red) or ADX < 20 (trend weakens).
# Uses Williams Alligator (SMMA-based) for trend identification, ADX for trend strength, volume for confirmation.
# Target: 60-120 total trades over 4 years (15-30/year) to avoid fee drag while capturing trends.

name = "4h_WilliamsAlligator_1dADX_Volume"
timeframe = "4h"
leverage = 1.0

def smma(data, period):
    """Smoothed Moving Average (SMMA) used in Williams Alligator"""
    if len(data) < period:
        return np.full_like(data, np.nan, dtype=float)
    result = np.full_like(data, np.nan, dtype=float)
    # First value is simple average
    result[period-1] = np.mean(data[:period])
    # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CURRENT_VALUE) / period
    for i in range(period, len(data)):
        result[i] = (result[i-1] * (period-1) + data[i]) / period
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Williams Alligator on 4h: Jaw (13,8), Teeth (8,5), Lips (5,3)
    # All are SMMA with different periods and shifts
    jaw = smma(high, 13)  # Blue line
    jaw = np.roll(jaw, 8)  # Shifted by 8 bars
    
    teeth = smma(high, 8)   # Red line
    teeth = np.roll(teeth, 5)  # Shifted by 5 bars
    
    lips = smma(high, 5)    # Green line
    lips = np.roll(lips, 3)  # Shifted by 3 bars
    
    # 1d ADX for trend strength filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX components
    plus_dm = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    minus_dm = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Smoothed values
    period = 14
    atr = np.zeros_like(high_1d)
    atr[period] = np.mean(tr[:period])
    for i in range(period+1, len(tr)):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
    
    # Avoid division by zero
    atr_safe = np.where(atr == 0, 1e-10, atr)
    
    plus_di = 100 * np.convolve(plus_dm, np.ones(period)/period, mode='same') / atr_safe
    minus_di = 100 * np.convolve(minus_dm, np.ones(period)/period, mode='same') / atr_safe
    
    dx = np.zeros_like(plus_di)
    dx = np.where((plus_di + minus_di) != 0, 
                  100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    
    adx = np.zeros_like(dx)
    adx[2*period-1] = np.mean(dx[:2*period-1]) if len(dx) >= 2*period-1 else 0
    for i in range(2*period, len(dx)):
        adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
    
    # Align ADX to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma20 = np.convolve(volume, np.ones(20)/20, mode='same')
    # Handle edges for convolution
    vol_ma20[:10] = vol_ma20[10]
    vol_ma20[-10:] = vol_ma20[-11]
    volume_filter = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Sufficient warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(lips[i]) or np.isnan(teeth[i]) or np.isnan(jaw[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: bullish Alligator alignment (green > red > blue), strong trend, volume
            bullish_align = (lips[i] > teeth[i]) and (teeth[i] > jaw[i])
            long_cond = bullish_align and (adx_aligned[i] > 25) and volume_filter[i]
            
            # Short conditions: bearish Alligator alignment (green < red < blue), strong trend, volume
            bearish_align = (lips[i] < teeth[i]) and (teeth[i] < jaw[i])
            short_cond = bearish_align and (adx_aligned[i] > 25) and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Alligator crosses bearish (green < red) OR trend weakens (ADX < 20)
            bearish_cross = lips[i] < teeth[i]
            weak_trend = adx_aligned[i] < 20
            if bearish_cross or weak_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Alligator crosses bullish (green > red) OR trend weakens (ADX < 20)
            bullish_cross = lips[i] > teeth[i]
            weak_trend = adx_aligned[i] < 20
            if bullish_cross or weak_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals