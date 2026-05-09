#!/usr/bin/env python3
# Hypothesis: 4h Williams Alligator + Elder Ray with volume confirmation and ADX filter
# Long when green line > red line (bullish alignment), Elder Ray bull power > 0, and ADX > 25
# Short when red line > green line (bearish alignment), Elder Ray bear power < 0, and ADX > 25
# Exit when Alligator lines converge (|green - red| < 0.1 * price) or Elder Ray power reverses
# Uses Alligator for trend alignment, Elder Ray for momentum, ADX for trend strength, volume for conviction
# Designed to capture strong trends in both bull and bear markets with controlled frequency
# Target: 60-120 total trades over 4 years (15-30/year) with size 0.25

name = "4h_Alligator_ElderRay_PowerTrend"
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
    
    # Calculate 1d Williams Alligator (Jaw, Teeth, Lips)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Alligator lines: Jaw (13,8), Teeth (8,5), Lips (5,3) - SMMA with offset
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    def smoothed_ma(data, period):
        sma = np.full_like(data, np.nan, dtype=float)
        if len(data) >= period:
            sma[period-1] = np.mean(data[:period])
            for i in range(period, len(data)):
                sma[i] = (sma[i-1] * (period-1) + data[i]) / period
        return sma
    
    jaw = smoothed_ma(close_1d, 13)  # Blue line
    teeth = smoothed_ma(close_1d, 8)  # Red line
    lips = smoothed_ma(close_1d, 5)   # Green line
    
    # Align Alligator lines to 4h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Calculate Elder Ray Power (13-period EMA)
    close_s = pd.Series(close)
    ema13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Calculate ADX (14-period)
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    tr = np.maximum(high[1:] - low[1:], np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])))
    atr = np.zeros_like(tr)
    for i in range(len(tr)):
        if i < 13:
            atr[i] = np.mean(tr[:i+1]) if i > 0 else tr[0]
        else:
            atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    atr = np.concatenate([np.array([np.nan]), atr])  # align with original indices
    
    plus_di = 100 * (pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr)
    minus_di = 100 * (pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_confirm = volume > (1.5 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for indicator calculation
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or np.isnan(adx[i]) or np.isnan(vol_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: lips > teeth (bullish alignment), bull power > 0, ADX > 25, volume confirmation
            if (lips_aligned[i] > teeth_aligned[i] and 
                bull_power[i] > 0 and 
                adx[i] > 25 and 
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: teeth > lips (bearish alignment), bear power < 0, ADX > 25, volume confirmation
            elif (teeth_aligned[i] > lips_aligned[i] and 
                  bear_power[i] < 0 and 
                  adx[i] > 25 and 
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: convergence of Alligator lines or bear power turns positive
            lips_teeth_diff = np.abs(lips_aligned[i] - teeth_aligned[i])
            convergence_threshold = 0.1 * close[i]
            if (lips_teeth_diff < convergence_threshold) or (bear_power[i] > 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: convergence of Alligator lines or bull power turns negative
            lips_teeth_diff = np.abs(lips_aligned[i] - teeth_aligned[i])
            convergence_threshold = 0.1 * close[i]
            if (lips_teeth_diff < convergence_threshold) or (bull_power[i] < 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals