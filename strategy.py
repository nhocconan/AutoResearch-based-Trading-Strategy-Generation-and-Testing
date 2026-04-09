#!/usr/bin/env python3
# 6h_1d_adx_ema_trend_follow_v1
# Hypothesis: Trend following on 6h using EMA(21) for direction and ADX(14) for strength, filtered by 1d EMA(50) trend.
# Enter long when 6h EMA(21) > EMA(50) AND ADX > 25 AND 1d EMA(50) > prior 1d EMA(50).
# Enter short when 6h EMA(21) < EMA(50) AND ADX > 25 AND 1d EMA(50) < prior 1d EMA(50).
# Exit when EMA cross reverses or ADX falls below 20.
# Designed to capture strong trends while avoiding choppy markets, works in both bull and bear via ADX filter.
# Target: 60-100 total trades over 4 years (15-25/year) with strict trend strength requirements.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_adx_ema_trend_follow_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Calculate EMA(21) and EMA(50) for 6h
    def ema(arr, period):
        result = np.full(n, np.nan)
        multiplier = 2.0 / (period + 1)
        sum_val = 0.0
        count = 0
        for i in range(n):
            if np.isnan(arr[i]):
                continue
            if count == 0:
                sum_val = arr[i]
            else:
                sum_val = arr[i] * multiplier + sum_val * (1 - multiplier)
            count += 1
            if count >= period:
                result[i] = sum_val
        return result
    
    ema_21 = ema(close, 21)
    ema_50 = ema(close, 50)
    
    # Calculate ADX(14)
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
        tr[0] = high[0] - low[0]  # First period
        
        # Directional Movement
        dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), np.maximum(high - np.roll(high, 1), 0), 0)
        dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), np.maximum(np.roll(low, 1) - low, 0), 0)
        dm_plus[0] = 0
        dm_minus[0] = 0
        
        # Smoothed values
        def smooth(arr, period):
            result = np.full(n, np.nan)
            if n < period:
                return result
            # Initial average
            result[period-1] = np.nansum(arr[1:period]) if period > 1 else arr[0]
            # Wilder smoothing
            for i in range(period, n):
                if np.isnan(result[i-1]):
                    result[i] = arr[i]
                else:
                    result[i] = (result[i-1] * (period - 1) + arr[i]) / period
            return result
        
        tr_smooth = smooth(tr, period)
        dm_plus_smooth = smooth(dm_plus, period)
        dm_minus_smooth = smooth(dm_minus, period)
        
        # Directional Indicators
        di_plus = np.where(tr_smooth != 0, 100 * dm_plus_smooth / tr_smooth, 0)
        di_minus = np.where(tr_smooth != 0, 100 * dm_minus_smooth / tr_smooth, 0)
        
        # DX
        dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
        
        # ADX
        adx = smooth(dx, period)
        return adx
    
    adx = calculate_adx(high, low, close, 14)
    
    # Load 1d data ONCE before loop for EMA(50) trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate EMA(50) on 1d close
    close_1d = df_1d['close'].values
    ema_50_1d = ema(close_1d, 50)
    
    # Align 1d EMA to 6h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if np.isnan(ema_21[i]) or np.isnan(ema_50[i]) or np.isnan(adx[i]) or np.isnan(ema_50_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        # 1d EMA trend direction (rising/falling)
        ema_1d_rising = ema_50_1d_aligned[i] > ema_50_1d_aligned[i-1] if i > 0 else False
        ema_1d_falling = ema_50_1d_aligned[i] < ema_50_1d_aligned[i-1] if i > 0 else False
        
        if position == 1:  # Long position
            # Exit: EMA cross down OR ADX weak (<20) OR 1d trend turns down
            if ema_21[i] < ema_50[i] or adx[i] < 20 or not ema_1d_rising:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: EMA cross up OR ADX weak (<20) OR 1d trend turns up
            if ema_21[i] > ema_50[i] or adx[i] < 20 or not ema_1d_falling:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: EMA21 > EMA50 AND ADX > 25 AND 1d EMA rising
            if ema_21[i] > ema_50[i] and adx[i] > 25 and ema_1d_rising:
                position = 1
                signals[i] = 0.25
            # Enter short: EMA21 < EMA50 AND ADX > 25 AND 1d EMA falling
            elif ema_21[i] < ema_50[i] and adx[i] > 25 and ema_1d_falling:
                position = -1
                signals[i] = -0.25
    
    return signals