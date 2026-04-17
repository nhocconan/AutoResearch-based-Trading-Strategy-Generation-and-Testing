#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray Index (Bull/Bear Power) + 1d ADX/EMA Regime Filter.
Long when Bull Power > 0 and Bear Power < 0 in trending market (ADX > 25) and price above 1d EMA50.
Short when Bear Power > 0 and Bull Power < 0 in trending market (ADX > 25) and price below 1d EMA50.
Exit when power diverges or regime changes to ranging (ADX < 20).
Uses 1d for ADX/EMA50 regime, 6h for Elder Ray calculation.
Target: 50-150 total trades over 4 years (12-37/year).
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
    
    # Get 1d data for ADX and EMA50 regime filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ADX (14-period)
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr = np.zeros_like(close)
        for i in range(1, len(close)):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Directional Movement
        dm_plus = np.zeros_like(close)
        dm_minus = np.zeros_like(close)
        for i in range(1, len(close)):
            up_move = high[i] - high[i-1]
            down_move = low[i-1] - low[i]
            dm_plus[i] = up_move if up_move > down_move and up_move > 0 else 0
            dm_minus[i] = down_move if down_move > up_move and down_move > 0 else 0
        
        # Smoothed TR, DM+
        tr_period = np.zeros_like(close)
        dm_plus_period = np.zeros_like(close)
        dm_minus_period = np.zeros_like(close)
        
        # Initial values (simple average)
        if len(close) >= period + 1:
            tr_period[period] = np.sum(tr[1:period+1])
            dm_plus_period[period] = np.sum(dm_plus[1:period+1])
            dm_minus_period[period] = np.sum(dm_minus[1:period+1])
            
            # Wilder smoothing
            for i in range(period+1, len(close)):
                tr_period[i] = tr_period[i-1] - (tr_period[i-1] / period) + tr[i]
                dm_plus_period[i] = dm_plus_period[i-1] - (dm_plus_period[i-1] / period) + dm_plus[i]
                dm_minus_period[i] = dm_minus_period[i-1] - (dm_minus_period[i-1] / period) + dm_minus[i]
        
        # Directional Indicators
        di_plus = np.zeros_like(close)
        di_minus = np.zeros_like(close)
        dx = np.zeros_like(close)
        
        for i in range(period, len(close)):
            if tr_period[i] != 0:
                di_plus[i] = 100 * (dm_plus_period[i] / tr_period[i])
                di_minus[i] = 100 * (dm_minus_period[i] / tr_period[i])
                if (di_plus[i] + di_minus[i]) != 0:
                    dx[i] = 100 * abs(di_plus[i] - di_minus[i]) / (di_plus[i] + di_minus[i])
        
        # ADX (smoothed DX)
        adx = np.zeros_like(close)
        if len(close) >= 2 * period:
            adx[2*period-1] = np.mean(dx[period:2*period])
            for i in range(2*period, len(close)):
                adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        return adx
    
    # Calculate 1d EMA50
    def calculate_ema(data, period):
        ema = np.zeros_like(data)
        if len(data) >= period:
            multiplier = 2 / (period + 1)
            ema[period-1] = np.mean(data[:period])
            for i in range(period, len(data)):
                ema[i] = (data[i] * multiplier) + (ema[i-1] * (1 - multiplier))
        return ema
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    ema50_1d = calculate_ema(close_1d, 50)
    
    # Calculate 6h Elder Ray Index (Bull Power = High - EMA13, Bear Power = Low - EMA13)
    def calculate_ema_fast(data, period):
        ema = np.zeros_like(data)
        if len(data) >= period:
            multiplier = 2 / (period + 1)
            ema[period-1] = np.mean(data[:period])
            for i in range(period, len(data)):
                ema[i] = (data[i] * multiplier) + (ema[i-1] * (1 - multiplier))
        return ema
    
    ema13 = calculate_ema_fast(close, 13)
    bull_power = high - ema13  # Bull Power = High - EMA
    bear_power = low - ema13   # Bear Power = Low - EMA
    
    # Align 1d indicators to 6h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(adx_1d_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i])):
            signals[i] = 0.0
            continue
        
        adx = adx_1d_aligned[i]
        ema50 = ema50_1d_aligned[i]
        bull = bull_power[i]
        bear = bear_power[i]
        price = close[i]
        
        # Regime filters
        is_trending = adx > 25          # Strong trend
        is_ranging = adx < 20           # Weak trend/ranging
        price_above_ema50 = price > ema50
        price_below_ema50 = price < ema50
        
        # Elder Ray conditions
        bull_strong = bull > 0          # Bull Power positive
        bear_strong = bear > 0          # Bear Power positive
        bull_weak = bull < 0            # Bull Power negative
        bear_weak = bear < 0            # Bear Power negative
        
        if position == 0:
            # Long: Bull Power > 0, Bear Power < 0, trending, price above EMA50
            if bull_strong and bear_weak and is_trending and price_above_ema50:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power > 0, Bull Power < 0, trending, price below EMA50
            elif bear_strong and bull_weak and is_trending and price_below_ema50:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Power diverges OR regime turns ranging
            if not (bull_strong and bear_weak) or is_ranging:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Power diverges OR regime turns ranging
            if not (bear_strong and bull_weak) or is_ranging:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_1dADXEMA_Regime"
timeframe = "6h"
leverage = 1.0