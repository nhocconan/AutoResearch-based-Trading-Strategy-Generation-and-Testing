#!/usr/bin/env python3
# Hypothesis: 6h Elder Ray Power + 1d ADX Trend Filter + Volume Spike
# Long when Bull Power > 0, Bear Power < 0, 1d ADX > 25 (trending), and volume > 2x average
# Short when Bear Power > 0, Bull Power < 0, 1d ADX > 25, and volume > 2x average
# Exit when Elder Power signals weaken or ADX weakens (<20)
# Uses Elder Ray to measure bull/bear power via EMA, ADX for trend strength, volume for conviction
# Designed to capture strong trending moves in both bull and bear markets with controlled frequency
# Target: 80-140 total trades over 4 years (20-35/year) with size 0.25

name = "6h_ElderRay_Power_ADXTrend_Volume"
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
    
    # Calculate 1d EMA13 for Elder Ray
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate EMA13 for Elder Ray
    ema13_1d = pd.Series(df_1d['close']).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray components: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power_1d = df_1d['high'].values - ema13_1d
    bear_power_1d = df_1d['low'].values - ema13_1d
    
    # Calculate 1d ADX for trend strength
    # Calculate +DM, -DM, TR
    plus_dm = np.where((df_1d['high'].values[1:] - df_1d['high'].values[:-1]) > 
                       (df_1d['low'].values[:-1] - df_1d['low'].values[1:]), 
                       np.maximum(df_1d['high'].values[1:] - df_1d['high'].values[:-1], 0), 0)
    minus_dm = np.where((df_1d['low'].values[:-1] - df_1d['low'].values[1:]) > 
                        (df_1d['high'].values[1:] - df_1d['high'].values[:-1]), 
                        np.maximum(df_1d['low'].values[:-1] - df_1d['low'].values[1:], 0), 0)
    
    # Handle first element
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    
    # True Range
    tr1 = df_1d['high'].values - df_1d['low'].values
    tr2 = np.abs(df_1d['high'].values - np.concatenate([[df_1d['close'].values[0]], df_1d['close'].values[:-1]]))
    tr3 = np.abs(df_1d['low'].values - np.concatenate([[df_1d['close'].values[0]], df_1d['close'].values[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Smooth with Wilder's smoothing (alpha = 1/period)
    def wilders_smooth(data, period):
        result = np.zeros_like(data)
        result[period-1] = np.mean(data[:period])
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    period = 14
    atr_1d = wilders_smooth(tr, period)
    plus_di_1d = 100 * wilders_smooth(plus_dm, period) / atr_1d
    minus_di_1d = 100 * wilders_smooth(minus_dm, period) / atr_1d
    dx_1d = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d)
    adx_1d = wilders_smooth(dx_1d, period)
    
    # Align 1d indicators to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Volume confirmation: current volume > 2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_confirm = volume > (2.0 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for ADX calculation
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(vol_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: Bull Power > 0, Bear Power < 0, ADX > 25 (strong trend), volume spike
            if (bull_power_aligned[i] > 0 and 
                bear_power_aligned[i] < 0 and 
                adx_aligned[i] > 25 and 
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: Bear Power > 0, Bull Power < 0, ADX > 25, volume spike
            elif (bear_power_aligned[i] > 0 and 
                  bull_power_aligned[i] < 0 and 
                  adx_aligned[i] > 25 and 
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Elder Power signals weaken or ADX weakens
            if (bull_power_aligned[i] <= 0 or 
                bear_power_aligned[i] >= 0 or 
                adx_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Elder Power signals weaken or ADX weakens
            if (bear_power_aligned[i] <= 0 or 
                bull_power_aligned[i] >= 0 or 
                adx_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals