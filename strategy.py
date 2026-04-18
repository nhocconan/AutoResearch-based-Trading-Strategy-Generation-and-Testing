#!/usr/bin/env python3
"""
4h Camarilla Pivot Breakout with Volume Confirmation and Choppiness Filter
Hypothesis: Camarilla pivot levels (H3/L3) act as strong support/resistance. 
Breakouts with volume confirmation indicate institutional participation. 
Choppiness filter avoids whipsaws in sideways markets. Works in both bull 
and bear markets by following breakout direction regardless of trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_highest(arr, period):
    """Calculate rolling highest"""
    if len(arr) < period:
        return np.full_like(arr, np.nan)
    result = np.full(len(arr), np.nan)
    for i in range(period-1, len(arr)):
        result[i] = np.max(arr[i-period+1:i+1])
    return result

def calculate_lowest(arr, period):
    """Calculate rolling lowest"""
    if len(arr) < period:
        return np.full_like(arr, np.nan)
    result = np.full(len(arr), np.nan)
    for i in range(period-1, len(arr)):
        result[i] = np.min(arr[i-period+1:i+1])
    return result

def calculate_rsi(close, period=14):
    """Calculate RSI with proper handling"""
    if len(close) < period + 1:
        return np.full_like(close, np.nan)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(close)
    avg_loss = np.zeros_like(close)
    
    avg_gain[period] = np.mean(gain[:period])
    avg_loss[period] = np.mean(loss[:period])
    
    for i in range(period + 1, len(close)):
        avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
        avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_atr(high, low, close, period=14):
    """Calculate Average True Range"""
    if len(high) < period:
        return np.full_like(high, np.nan)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First TR is just high-low
    
    atr = np.zeros_like(high)
    atr[0] = tr[0]
    for i in range(1, len(tr)):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla formulas using previous day's range
    range_1d = high_1d - low_1d
    camarilla_h3 = close_1d + (range_1d * 1.1 / 6)
    camarilla_l3 = close_1d - (range_1d * 1.1 / 6)
    camarilla_h4 = close_1d + (range_1d * 1.1 / 2)
    camarilla_l4 = close_1d - (range_1d * 1.1 / 2)
    
    # Align to 4h timeframe (use previous day's levels)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = np.zeros_like(volume)
    for i in range(len(volume)):
        if i < 20:
            vol_ma[i] = np.mean(volume[max(0, i-19):i+1]) if i >= 0 else volume[i]
        else:
            vol_ma[i] = np.mean(volume[i-19:i+1])
    vol_spike = volume > (vol_ma * 1.5)
    
    # Choppiness filter: avoid trading in choppy markets
    # Calculate Chop using ATR and true range over 14 periods
    atr_val = calculate_atr(high, low, close, 14)
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    
    sum_tr = np.zeros_like(atr_val)
    for i in range(len(sum_tr)):
        if i < 14:
            sum_tr[i] = np.sum(tr[max(0, i-13):i+1])
        else:
            sum_tr[i] = np.sum(tr[i-13:i+1])
    
    # Avoid division by zero
    max_hh = calculate_highest(high, 14)
    min_ll = calculate_lowest(low, 14)
    denominator = max_hh - min_ll
    denominator = np.where(denominator == 0, 1e-10, denominator)
    
    chop = np.where(denominator != 0, 100 * np.log10(sum_tr / denominator) / np.log10(14), 50)
    chop_threshold = 61.8  # Above this is choppy/choppy market
    trending_market = chop < chop_threshold  # Only trade when NOT choppy
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(vol_ma[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long breakout: price breaks above H3 with volume in trending market
            if (close[i] > camarilla_h3_aligned[i] and 
                vol_spike[i] and 
                trending_market[i]):
                signals[i] = 0.25
                position = 1
            # Short breakdown: price breaks below L3 with volume in trending market
            elif (close[i] < camarilla_l3_aligned[i] and 
                  vol_spike[i] and 
                  trending_market[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns below H3 or volatility spike ends
            if close[i] < camarilla_h3_aligned[i] or not vol_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns above L3 or volatility spike ends
            if close[i] > camarilla_l3_aligned[i] or not vol_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_H3L3_Breakout_Volume_ChopFilter"
timeframe = "4h"
leverage = 1.0