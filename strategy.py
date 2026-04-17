#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray Index + 1d ADX/EMA Regime Filter.
Long when Bull Power > 0 and Bear Power < 0 in trending market (ADX > 25) with price above 1d EMA50.
Short when Bear Power < 0 and Bull Power > 0 in trending market (ADX > 25) with price below 1d EMA50.
Exit when power diverges or regime turns ranging (ADX < 20).
Uses 13-period Elder Ray for responsiveness, 1d ADX(14) and EMA(50) for regime.
Target: 50-150 total trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for ADX and EMA regime filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 13-period Elder Ray (Bull Power = High - EMA13, Bear Power = Low - EMA13)
    def calculate_ema(arr, period):
        ema = np.full_like(arr, np.nan)
        multiplier = 2 / (period + 1)
        ema[period-1] = np.mean(arr[:period])
        for i in range(period, len(arr)):
            ema[i] = (arr[i] - ema[i-1]) * multiplier + ema[i-1]
        return ema
    
    ema13 = calculate_ema(close, 13)
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Calculate 1d ADX (14-period)
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr = np.zeros_like(high)
        for i in range(1, len(high)):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Directional Movement
        dm_plus = np.zeros_like(high)
        dm_minus = np.zeros_like(high)
        for i in range(1, len(high)):
            up_move = high[i] - high[i-1]
            down_move = low[i-1] - low[i]
            dm_plus[i] = up_move if up_move > down_move and up_move > 0 else 0
            dm_minus[i] = down_move if down_move > up_move and down_move > 0 else 0
        
        # Smoothed TR, DM+ (Wilder's smoothing)
        atr = np.zeros_like(high)
        dm_plus_smooth = np.zeros_like(high)
        dm_minus_smooth = np.zeros_like(high)
        atr[period] = np.mean(tr[1:period+1])
        dm_plus_smooth[period] = np.mean(dm_plus[1:period+1])
        dm_minus_smooth[period] = np.mean(dm_minus[1:period+1])
        for i in range(period+1, len(high)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
            dm_plus_smooth[i] = (dm_plus_smooth[i-1] * (period-1) + dm_plus[i]) / period
            dm_minus_smooth[i] = (dm_minus_smooth[i-1] * (period-1) + dm_minus[i]) / period
        
        # Directional Indicators
        di_plus = np.zeros_like(high)
        di_minus = np.zeros_like(high)
        for i in range(period, len(high)):
            if atr[i] != 0:
                di_plus[i] = (dm_plus_smooth[i] / atr[i]) * 100
                di_minus[i] = (dm_minus_smooth[i] / atr[i]) * 100
        
        # DX and ADX
        dx = np.zeros_like(high)
        for i in range(period, len(high)):
            if di_plus[i] + di_minus[i] != 0:
                dx[i] = abs(di_plus[i] - di_minus[i]) / (di_plus[i] + di_minus[i]) * 100
        
        adx = np.zeros_like(high)
        adx[2*period-1] = np.mean(dx[period:2*period])
        for i in range(2*period, len(high)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    
    # Calculate 1d EMA50
    ema50_1d = calculate_ema(close_1d, 50)
    
    # Align 1d indicators to 6h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(adx_1d_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or
            np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        adx_val = adx_1d_aligned[i]
        ema50 = ema50_1d_aligned[i]
        bull = bull_power[i]
        bear = bear_power[i]
        
        # Regime: ADX > 25 = trending (good for trend following)
        is_trending = adx_val > 25
        # Exit regime: ADX < 20 = ranging (avoid false signals)
        is_ranging = adx_val < 20
        
        if position == 0:
            # Long: Bull Power > 0 and Bear Power < 0 in trending market with price above EMA50
            if bull > 0 and bear < 0 and is_trending and price > ema50:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 and Bull Power > 0 in trending market with price below EMA50
            elif bear < 0 and bull > 0 and is_trending and price < ema50:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: power diverges OR regime turns ranging
            if bull <= 0 or bear >= 0 or is_ranging:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: power diverges OR regime turns ranging
            if bear >= 0 or bull <= 0 or is_ranging:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_1dADXEMA_Regime"
timeframe = "6h"
leverage = 1.0