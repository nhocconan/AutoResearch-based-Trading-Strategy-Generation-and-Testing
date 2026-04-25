#!/usr/bin/env python3
"""
12h_WilliamsAlligator_ElderRay_Trend_Filter
Hypothesis: 12h Williams Alligator (JAW/TEETH/LIPS) combined with Elder Ray (Bull/Bear Power) 
to identify strong trends with momentum confirmation. Uses 1d HTF for regime filter (ADX>25) 
to avoid whipsaws. Long when price > LIPS, Bull Power > 0, and ADX>25. Short when price < JAW, 
Bear Power < 0, and ADX>25. Exit on opposite Alligator line cross or ADX<20. 
Designed for low trade frequency (12-37/year) with discrete sizing (0.25) to minimize fees.
Works in bull via trend continuation, in bear via filtered short signals during strong downtrends.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 12h data for Williams Alligator and Elder Ray calculations
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 13:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Williams Alligator: SMAs of median price
    # JAW: 13-period SMMA shifted 8 bars
    # TEETH: 8-period SMMA shifted 5 bars  
    # LIPS: 5-period SMMA shifted 3 bars
    median_12h = (high_12h + low_12h) / 2
    
    def smma(arr, period):
        """Smoothed Moving Average"""
        result = np.full_like(arr, np.nan)
        if len(arr) < period:
            return result
        # First value is simple SMA
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CURRENT) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw_12h = smma(median_12h, 13)
    teeth_12h = smma(median_12h, 8)
    lips_12h = smma(median_12h, 5)
    
    # Shift the lines as per Alligator specification
    jaw_12h = np.roll(jaw_12h, 8)
    teeth_12h = np.roll(teeth_12h, 5)
    lips_12h = np.roll(lips_12h, 3)
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema_13_12h = pd.Series(close_12h).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power_12h = high_12h - ema_13_12h
    bear_power_12h = low_12h - ema_13_12h
    
    # Align Alligator and Elder Ray to original timeframe
    jaw_12h_aligned = align_htf_to_ltf(prices, df_12h, jaw_12h)
    teeth_12h_aligned = align_htf_to_ltf(prices, df_12h, teeth_12h)
    lips_12h_aligned = align_htf_to_ltf(prices, df_12h, lips_12h)
    bull_power_12h_aligned = align_htf_to_ltf(prices, df_12h, bull_power_12h)
    bear_power_12h_aligned = align_htf_to_ltf(prices, df_12h, bear_power_12h)
    
    # Get 1d data for ADX regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # ADX calculation (14-period)
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])  # align with original index
        
        # Directional Movement
        dm_plus = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), 
                           np.maximum(high[1:] - high[:-1], 0), 0)
        dm_plus = np.concatenate([[np.nan], dm_plus])
        dm_minus = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), 
                            np.maximum(low[:-1] - low[1:], 0), 0)
        dm_minus = np.concatenate([[np.nan], dm_minus])
        
        # Smoothed TR, DM+
        tr_period = pd.Series(tr).ewm(span=period, adjust=False, min_periods=period).mean().values
        dm_plus_period = pd.Series(dm_plus).ewm(span=period, adjust=False, min_periods=period).mean().values
        dm_minus_period = pd.Series(dm_minus).ewm(span=period, adjust=False, min_periods=period).mean().values
        
        # Directional Indicators
        di_plus = 100 * dm_plus_period / tr_period
        di_minus = 100 * dm_minus_period / tr_period
        
        # DX and ADX
        dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
        adx = pd.Series(dx).ewm(span=period, adjust=False, min_periods=period).mean().values
        return adx
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for calculations
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaw_12h_aligned[i]) or np.isnan(lips_12h_aligned[i]) or 
            np.isnan(bull_power_12h_aligned[i]) or np.isnan(bear_power_12h_aligned[i]) or
            np.isnan(adx_1d_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        if position == 0:
            # Long: price > LIPS, Bull Power > 0, ADX > 25 (strong uptrend)
            long_signal = (close[i] > lips_12h_aligned[i]) and \
                         (bull_power_12h_aligned[i] > 0) and \
                         (adx_1d_aligned[i] > 25)
            # Short: price < JAW, Bear Power < 0, ADX > 25 (strong downtrend)
            short_signal = (close[i] < jaw_12h_aligned[i]) and \
                          (bear_power_12h_aligned[i] < 0) and \
                          (adx_1d_aligned[i] > 25)
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit conditions: price < JAW or ADX < 20 (trend weakening)
            exit_signal = (close[i] < jaw_12h_aligned[i]) or (adx_1d_aligned[i] < 20)
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit conditions: price > LIPS or ADX < 20 (trend weakening)
            exit_signal = (close[i] > lips_12h_aligned[i]) or (adx_1d_aligned[i] < 20)
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_WilliamsAlligator_ElderRay_Trend_Filter"
timeframe = "12h"
leverage = 1.0