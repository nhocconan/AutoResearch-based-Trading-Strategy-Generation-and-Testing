#!/usr/bin/env python3
"""
6h_1d_Alligator_Range_Filter_v1
Hypothesis: Bill Williams Alligator identifies trending vs ranging markets. In range (JAW>TEETH>LIPS or reverse), fade price extremes at Bollinger Bands (20,2) on 6h. In trend (JAW<TEETH<LIPS or reverse), follow Alligator crossovers. Uses 1d Alligator for regime, 6h for entries. Avoids whipsaw in chop, catches trends. Works in bull (trend follow) and bear (range fade). Target 15-30 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_Alligator_Range_Filter_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1D data for Alligator regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === 1D ALLIGATOR (13,8,5 SMMA) ===
    def smma(arr, period):
        """Smoothed Moving Average"""
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        res = np.full(len(arr), np.nan)
        sma = np.mean(arr[:period])
        res[period-1] = sma
        for i in range(period, len(arr)):
            res[i] = (res[i-1] * (period-1) + arr[i]) / period
        return res
    
    ma13 = smma(df_1d['close'].values, 13)
    ma8 = smma(df_1d['close'].values, 8)
    ma5 = smma(df_1d['close'].values, 5)
    
    # Align Alligator lines to 6h
    jaw_6h = align_htf_to_ltf(prices, df_1d, ma13)
    teeth_6h = align_htf_to_ltf(prices, df_1d, ma8)
    lips_6h = align_htf_to_ltf(prices, df_1d, ma5)
    
    # === 6H BOLLINGER BANDS (20, 2) ===
    if n < 20:
        return np.zeros(n)
    
    sma20 = np.full(n, np.nan)
    std20 = np.full(n, np.nan)
    sum_close = np.sum(close[:20])
    sum_sq = np.sum(close[:20]**2)
    sma20[19] = sum_close / 20
    var20 = (sum_sq / 20) - (sma20[19]**2)
    std20[19] = np.sqrt(max(var20, 0))
    upper_bb = sma20 + 2 * std20
    lower_bb = sma20 - 2 * std20
    
    for i in range(20, n):
        sum_close = sum_close - close[i-20] + close[i]
        sum_sq = sum_sq - close[i-20]**2 + close[i]**2
        sma20[i] = sum_close / 20
        var20 = (sum_sq / 20) - (sma20[i]**2)
        std20[i] = np.sqrt(max(var20, 0))
        upper_bb[i] = sma20[i] + 2 * std20[i]
        lower_bb[i] = sma20[i] - 2 * std20[i]
    
    # === SIGNAL LOGIC ===
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any data invalid
        if (np.isnan(jaw_6h[i]) or np.isnan(teeth_6h[i]) or np.isnan(lips_6h[i]) or
            np.isnan(upper_bb[i]) or np.isnan(lower_bb[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine regime: trending or ranging
        # Trend: JAW < TEETH < LIPS (bull) or JAW > TEETH > LIPS (bear)
        # Range: JAW > TEETH > LIPS (bull alignment but prices in range) or JAW < TEETH < LIPS (bear alignment but ranging)
        # Actually: Alligator sleeping (all intertwined) = range; waking up (separated) = trend
        # Simpler: if lips > teeth > jaw OR lips < teeth < jaw = trending, else ranging
        lips_above_teeth = lips_6h[i] > teeth_6h[i]
        teeth_above_jaw = teeth_6h[i] > jaw_6h[i]
        lips_below_teeth = lips_6h[i] < teeth_6h[i]
        teeth_below_jaw = teeth_6h[i] < jaw_6h[i]
        
        trending = (lips_above_teeth and teeth_above_jaw) or (lips_below_teeth and teeth_below_jaw)
        ranging = not trending
        
        if ranging:
            # Fade Bollinger Band extremes
            long_entry = close[i] <= lower_bb[i]
            short_entry = close[i] >= upper_bb[i]
            long_exit = close[i] >= (sma20[i])  # Exit at middle BB
            short_exit = close[i] <= (sma20[i])
        else:
            # Follow Alligator crossover (Lips cross Teeth)
            # Need previous values for crossover
            if i == 50:
                prev_lips = lips_6h[i-1] if not np.isnan(lips_6h[i-1]) else lips_6h[i]
                prev_teeth = teeth_6h[i-1] if not np.isnan(teeth_6h[i-1]) else teeth_6h[i]
            else:
                prev_lips = lips_6h[i-1]
                prev_teeth = teeth_6h[i-1]
            
            # Bullish crossover: Lips crosses above Teeth
            bullish_cross = (prev_lips <= prev_teeth) and (lips_6h[i] > teeth_6h[i])
            # Bearish crossover: Lips crosses below Teeth
            bearish_cross = (prev_lips >= prev_teeth) and (lips_6h[i] < teeth_6h[i])
            
            long_entry = bullish_cross
            short_entry = bearish_cross
            # Exit when cross reverses
            long_exit = bearish_cross
            short_exit = bullish_cross
        
        # Signal logic
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals