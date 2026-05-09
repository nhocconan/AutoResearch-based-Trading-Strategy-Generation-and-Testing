#!/usr/bin/env python3
# 4h_TRIX_VolumeSpike_Regime
# Hypothesis: Uses TRIX momentum oscillator (12-period) combined with volume spikes and Choppiness Index regime filter.
# TRIX identifies momentum shifts, volume confirms breakout strength, and Choppiness Index (CI > 61.8) filters for range-bound conditions.
# Works in both bull and bear markets by capturing momentum reversals in ranging conditions.
# Target: 20-35 trades/year per symbol with strict entry conditions to avoid overtrading.

name = "4h_TRIX_VolumeSpike_Regime"
timeframe = "4h"
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
    
    # Get daily data for Choppiness Index and TRIX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate TRIX (12-period) on daily close
    # TRIX = EMA(EMA(EMA(close, 12), 12), 12) - 1 period percent change
    def ema(arr, period):
        result = np.full_like(arr, np.nan)
        if len(arr) < period:
            return result
        k = 2 / (period + 1)
        result[period-1] = np.mean(arr[0:period])
        for i in range(period, len(arr)):
            result[i] = arr[i] * k + result[i-1] * (1 - k)
        return result
    
    ema1 = ema(close_1d, 12)
    ema2 = ema(ema1, 12)
    ema3 = ema(ema2, 12)
    trix = np.full_like(close_1d, np.nan)
    valid = ~np.isnan(ema3)
    trix[valid] = (ema3[valid] - np.roll(ema3, 1)[valid]) / np.roll(ema3, 1)[valid] * 100
    
    # Calculate Choppiness Index (14-period) on daily data
    def choppiness_index(high_arr, low_arr, close_arr, period=14):
        ci = np.full_like(close_arr, np.nan)
        if len(close_arr) < period:
            return ci
        atr = np.full_like(close_arr, np.nan)
        for i in range(len(close_arr)):
            if i == 0:
                tr = high_arr[i] - low_arr[i]
            else:
                tr = max(high_arr[i] - low_arr[i], 
                         abs(high_arr[i] - close_arr[i-1]),
                         abs(low_arr[i] - close_arr[i-1]))
            if i < 1:
                atr[i] = tr
            else:
                atr[i] = (atr[i-1] * (period-1) + tr) / period
        
        for i in range(period-1, len(close_arr)):
            if np.isnan(atr[i]) or atr[i] == 0:
                continue
            highest_high = np.max(high_arr[i-period+1:i+1])
            lowest_low = np.min(low_arr[i-period+1:i+1])
            if highest_high == lowest_low:
                ci[i] = 50
            else:
                ci[i] = 100 * np.log10(atr[i] * period / (highest_high - lowest_low)) / np.log10(period)
        return ci
    
    ci = choppiness_index(high_1d, low_1d, close_1d, 14)
    
    # Align daily indicators to 4h timeframe
    trix_aligned = align_htf_to_ltf(prices, df_1d, trix)
    ci_aligned = align_htf_to_ltf(prices, df_1d, ci)
    
    # Volume filter: 4h volume / 20-period average volume
    vol_ma = np.full_like(volume, np.nan)
    if len(volume) >= 20:
        vol_ma[19] = np.mean(volume[0:20])
        for i in range(20, len(volume)):
            vol_ma[i] = (vol_ma[i-1] * 19 + volume[i]) / 20
    
    volume_ratio = np.full_like(volume, np.nan)
    valid_vol = (~np.isnan(vol_ma)) & (vol_ma != 0)
    volume_ratio[valid_vol] = volume[valid_vol] / vol_ma[valid_vol]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 1)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(trix_aligned[i]) or np.isnan(ci_aligned[i]) or np.isnan(volume_ratio[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Regime filter: Choppiness Index > 61.8 indicates ranging market (good for mean reversion)
        ranging_market = ci_aligned[i] > 61.8
        
        if position == 0:
            # Enter long: TRIX crosses above zero (bullish momentum) AND volume spike AND ranging market
            if i > 0 and trix_aligned[i-1] <= 0 and trix_aligned[i] > 0 and volume_ratio[i] > 2.0 and ranging_market:
                signals[i] = 0.25
                position = 1
            # Enter short: TRIX crosses below zero (bearish momentum) AND volume spike AND ranging market
            elif i > 0 and trix_aligned[i-1] >= 0 and trix_aligned[i] < 0 and volume_ratio[i] > 2.0 and ranging_market:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: TRIX crosses below zero OR Choppiness drops below 38.2 (trending market)
            if (i > 0 and trix_aligned[i-1] > 0 and trix_aligned[i] <= 0) or ci_aligned[i] < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: TRIX crosses above zero OR Choppiness drops below 38.2 (trending market)
            if (i > 0 and trix_aligned[i-1] < 0 and trix_aligned[i] >= 0) or ci_aligned[i] < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals