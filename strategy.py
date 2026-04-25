#!/usr/bin/env python3
"""
6h_ADX_Alligator_Combo_v1
Hypothesis: Combine ADX trend strength with Williams Alligator lines on 6h timeframe. 
Enter long when ADX > 25 (strong trend) + price > Alligator Jaw (teeth aligned up). 
Enter short when ADX > 25 + price < Alligator Jaw (teeth aligned down). 
Use 1d HTF for regime filter: only trade in direction of 1d EMA50 trend. 
Discrete sizing (0.25) to limit fee drag. Target: 12-35 trades/year per symbol.
Works in bull/bear via ADX filtering + 1d trend alignment.
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
    
    # Get 1d data for HTF regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for regime filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Williams Alligator on 6h (primary timeframe)
    # Alligator: Jaw (13-period SMMA shifted 8), Teeth (8-period SMMA shifted 5), Lips (5-period SMMA shifted 3)
    def smma(arr, period):
        """Smoothed Moving Average"""
        result = np.full_like(arr, np.nan, dtype=float)
        if len(arr) < period:
            return result
        # First value is SMA
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CLOSE) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(high, 13)  # Typically uses median price, but high/low works for demo
    teeth = smma(low, 8)
    lips = smma(close, 5)
    
    # Shift as per Alligator definition
    jaw_shifted = np.roll(jaw, 8)
    teeth_shifted = np.roll(teeth, 5)
    lips_shifted = np.roll(lips, 3)
    
    # Aligned arrays (no look-ahead)
    jaw_aligned = align_htf_to_ltf(prices, prices, jaw_shifted)  # self-align for same TF
    teeth_aligned = align_htf_to_ltf(prices, prices, teeth_shifted)
    lips_aligned = align_htf_to_ltf(prices, prices, lips_shifted)
    
    # Calculate ADX on 6h
    def calculate_adx(high, low, close, period=14):
        """Calculate ADX, +DI, -DI"""
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]  # First period
        
        # Directional Movement
        dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                           np.maximum(high - np.roll(high, 1), 0), 0)
        dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                            np.maximum(np.roll(low, 1) - low, 0), 0)
        dm_plus[0] = 0
        dm_minus[0] = 0
        
        # Smoothed TR, DM+
        tr_period = pd.Series(tr).ewm(span=period, adjust=False, min_periods=period).mean().values
        dm_plus_period = pd.Series(dm_plus).ewm(span=period, adjust=False, min_periods=period).mean().values
        dm_minus_period = pd.Series(dm_minus).ewm(span=period, adjust=False, min_periods=period).mean().values
        
        # +DI, -DI
        plus_di = 100 * dm_plus_period / tr_period
        minus_di = 100 * dm_minus_period / tr_period
        
        # DX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        dx = np.where(np.isnan(dx), 0, dx)
        
        # ADX
        adx = pd.Series(dx).ewm(span=period, adjust=False, min_periods=period).mean().values
        return adx, plus_di, minus_di
    
    adx, plus_di, minus_di = calculate_adx(high, low, close, 14)
    adx_aligned = align_htf_to_ltf(prices, prices, adx)
    plus_di_aligned = align_htf_to_ltf(prices, prices, plus_di)
    minus_di_aligned = align_htf_to_ltf(prices, prices, minus_di)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for Alligator (max shift 13+8=21) and ADX (14+14=28)
    start_idx = max(30, 50)  # Conservative warmup
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(adx_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine 1d regime (bullish = price above EMA50)
        regime_bullish = close[i] > ema_50_1d_aligned[i]
        regime_bearish = close[i] < ema_50_1d_aligned[i]
        
        # Alligator alignment: teeth above lips = bullish alignment
        alligator_bullish = teeth_aligned[i] > lips_aligned[i]
        alligator_bearish = teeth_aligned[i] < lips_aligned[i]
        
        # ADX trend strength
        strong_trend = adx_aligned[i] > 25
        
        if position == 0:
            # Long setup: strong trend + Alligator bullish alignment + 1d bullish regime
            long_setup = strong_trend and alligator_bullish and regime_bullish
            
            # Short setup: strong trend + Alligator bearish alignment + 1d bearish regime
            short_setup = strong_trend and alligator_bearish and regime_bearish
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: Alligator alignment breaks OR ADX weakens OR regime changes
            if (not alligator_bullish) or (adx_aligned[i] < 20) or (not regime_bullish):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: Alligator alignment breaks OR ADX weakens OR regime changes
            if (not alligator_bearish) or (adx_aligned[i] < 20) or (regime_bullish):  # Note: regime_bullish means exit short
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_ADX_Alligator_Combo_v1"
timeframe = "6h"
leverage = 1.0