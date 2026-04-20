#!/usr/bin/env python3
# 12h_Camarilla_R1S1_Breakout_VolumeTrend_Regime
# Hypothesis: Camarilla pivot levels (R1/S1) from daily timeframe provide institutional support/resistance.
# Breakouts above R1 or below S1 with volume confirmation and ADX > 25 trend filter capture momentum.
# Choppiness index regime filter avoids false breakouts in sideways markets (CHOP > 61.8 = range).
# Works in bull/bear by following institutional flow at key levels. Target: 15-35 trades/year.

name = "12h_Camarilla_R1S1_Breakout_VolumeTrend_Regime"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given high, low, close arrays."""
    pivot = (high + low + close) / 3
    range_val = high - low
    r1 = close + (range_val * 1.1 / 12)
    s1 = close - (range_val * 1.1 / 12)
    return r1, s1, pivot

def calculate_adx(high, low, close, period=14):
    """Calculate ADX using Wilder's smoothing."""
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), 
                       np.maximum(high[1:] - high[:-1], 0), 0)
    dm_minus = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), 
                        np.maximum(low[:-1] - low[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Wilder's smoothing (alpha = 1/period)
    def WilderSmooth(data, period):
        smoothed = np.full_like(data, np.nan)
        alpha = 1.0 / period
        for i in range(len(data)):
            if np.isnan(smoothed[i-1]):
                smoothed[i] = data[i]
            else:
                smoothed[i] = alpha * data[i] + (1 - alpha) * smoothed[i-1]
        return smoothed
    
    tr_smoothed = WilderSmooth(tr, period)
    dm_plus_smoothed = WilderSmooth(dm_plus, period)
    dm_minus_smoothed = WilderSmooth(dm_minus, period)
    
    # Directional Indicators
    di_plus = 100 * dm_plus_smoothed / tr_smoothed
    di_minus = 100 * dm_minus_smoothed / tr_smoothed
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    
    # ADX
    adx = WilderSmooth(dx, period)
    return adx

def calculate_choppiness(high, low, close, period=14):
    """Calculate Choppiness Index."""
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Sum of TRUE RANGE over period
    tr_sum = np.nansum(tr.reshape(-1, period), axis=1)
    tr_sum_padded = np.concatenate([np.full(period-1, np.nan), tr_sum])
    
    # Highest high and lowest low over period
    max_high = np.maximum.accumulate(high)
    min_low = np.minimum.accumulate(low)
    range_max = max_high - min_low
    
    # Chop = 100 * log10(tr_sum / range_max) / log10(period)
    chop = 100 * np.log10(tr_sum_padded / range_max) / np.log10(period)
    return chop

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Camarilla levels on daily data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    r1_1d, s1_1d, pivot_1d = calculate_camarilla(high_1d, low_1d, close_1d)
    
    # Align Camarilla levels to 12h timeframe
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # Calculate ADX on 12h data for trend strength
    adx = calculate_adx(high, low, close, period=14)
    
    # Calculate Choppiness Index on 12h data for regime filter
    chop = calculate_choppiness(high, low, close, period=14)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)  # Ensure indicators are calculated
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) or
            np.isnan(adx[i]) or np.isnan(chop[i]) or np.isnan(volume_filter[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: only trade when market is trending (CHOP <= 61.8)
        if chop[i] > 61.8:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R1 + ADX > 25 + volume confirmation
            if close[i] > r1_1d_aligned[i] and adx[i] > 25 and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 + ADX > 25 + volume confirmation
            elif close[i] < s1_1d_aligned[i] and adx[i] > 25 and volume_filter[i]:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price breaks below S1 or ADX weakens
            if close[i] < s1_1d_aligned[i] or adx[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price breaks above R1 or ADX weakens
            if close[i] > r1_1d_aligned[i] or adx[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals