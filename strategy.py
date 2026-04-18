#!/usr/bin/env python3
"""
1d Weekly Pivot Reversal with Volume Spike and Trend Filter
Hypothesis: Weekly pivot levels (S1/S2/R1/R2) act as key support/resistance on daily chart.
Price reversals from these levels with volume spikes indicate institutional defense.
Trend filter (weekly EMA) ensures alignment with higher timeframe momentum.
Works in both bull and bear markets by fading extremes in ranging conditions
and following trend when strong.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_ema(arr, period):
    """Calculate Exponential Moving Average"""
    if len(arr) < period:
        return np.full_like(arr, np.nan)
    ema = np.zeros_like(arr)
    multiplier = 2 / (period + 1)
    ema[0] = arr[0]
    for i in range(1, len(arr)):
        ema[i] = (arr[i] - ema[i-1]) * multiplier + ema[i-1]
    return ema

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
    
    # Get weekly data for pivot levels and trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot levels (standard floor trader pivots)
    high_w = df_1w['high'].values
    low_w = df_1w['low'].values
    close_w = df_1w['close'].values
    
    pivot = (high_w + low_w + close_w) / 3
    r1 = 2 * pivot - low_w
    s1 = 2 * pivot - high_w
    r2 = pivot + (high_w - low_w)
    s2 = pivot - (high_w - low_w)
    
    # Align weekly levels to daily timeframe (use previous week's levels)
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2)
    
    # Weekly EMA for trend filter (34-period)
    ema34_w = calculate_ema(close_w, 34)
    ema34_w_aligned = align_htf_to_ltf(prices, df_1w, ema34_w)
    
    # Daily RSI for overbought/oversold
    rsi = calculate_rsi(close, 14)
    
    # Volume spike: current volume > 2x 20-period average
    vol_ma = np.zeros_like(volume)
    for i in range(len(volume)):
        if i < 20:
            vol_ma[i] = np.mean(volume[max(0, i-19):i+1]) if i >= 0 else volume[i]
        else:
            vol_ma[i] = np.mean(volume[i-19:i+1])
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or np.isnan(ema34_w_aligned[i]) or
            np.isnan(rsi[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long setup: price near S1/S2 with bullish bias
            near_support = (abs(close[i] - s1_aligned[i]) < (high[i] - low[i]) * 0.5 or 
                           abs(close[i] - s2_aligned[i]) < (high[i] - low[i]) * 0.5)
            bullish_bias = close[i] > ema34_w_aligned[i]  # Above weekly EMA
            oversold = rsi[i] < 30
            
            if near_support and (bullish_bias or oversold) and vol_spike[i]:
                signals[i] = 0.25
                position = 1
            
            # Short setup: price near R1/R2 with bearish bias
            near_resistance = (abs(close[i] - r1_aligned[i]) < (high[i] - low[i]) * 0.5 or 
                              abs(close[i] - r2_aligned[i]) < (high[i] - low[i]) * 0.5)
            bearish_bias = close[i] < ema34_w_aligned[i]  # Below weekly EMA
            overbought = rsi[i] > 70
            
            if near_resistance and (bearish_bias or overbought) and vol_spike[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price reaches pivot, RSI overbought, or loses momentum
            if (close[i] > pivot_aligned[i] or rsi[i] > 70 or 
                close[i] < ema34_w_aligned[i] * 0.98):  # 2% below weekly EMA
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price reaches pivot, RSI oversold, or gains momentum
            if (close[i] < pivot_aligned[i] or rsi[i] < 30 or 
                close[i] > ema34_w_aligned[i] * 1.02):  # 2% above weekly EMA
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WeeklyPivot_RSI_VolumeSpike"
timeframe = "1d"
leverage = 1.0