#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and ATR-based position sizing.
Long when price breaks above 20-day high with 1w EMA50 uptrend.
Short when price breaks below 20-day low with 1w EMA50 downtrend.
Exit when price reverts to 10-day EMA or ATR stoploss hit.
Uses discrete position sizing (0.0, ±0.25) to minimize fee churn.
Target: 30-100 total trades over 4 years (7-25/year).
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
    
    # Get 1d data for Donchian channels and EMA10
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate 1d Donchian channels (20-period)
    def calculate_donchian(high, low, period=20):
        upper = np.full_like(high, np.nan)
        lower = np.full_like(high, np.nan)
        for i in range(period-1, len(high)):
            upper[i] = np.max(high[i-period+1:i+1])
            lower[i] = np.min(low[i-period+1:i+1])
        return upper, lower
    
    donch_upper_1d, donch_lower_1d = calculate_donchian(high_1d, low_1d, 20)
    
    # Calculate 1d EMA10 for exit
    def calculate_ema(values, period):
        ema = np.full_like(values, np.nan)
        if len(values) >= period:
            multiplier = 2 / (period + 1)
            ema[period-1] = np.mean(values[:period])
            for i in range(period, len(values)):
                ema[i] = (values[i] * multiplier) + (ema[i-1] * (1 - multiplier))
        return ema
    
    ema10_1d = calculate_ema(close_1d, 10)
    
    # Calculate 1w EMA50 for trend filter
    ema50_1w = calculate_ema(close_1w, 50)
    
    # Calculate ATR(14) for stoploss
    def calculate_atr(high, low, close, period=14):
        tr = np.zeros_like(close)
        for i in range(1, len(close)):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        atr = np.zeros_like(close)
        if len(tr) >= period:
            atr[period] = np.mean(tr[1:period+1])
            for i in range(period+1, len(tr)):
                atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        return atr
    
    atr_1d = calculate_atr(high_1d, low_1d, close_1d, 14)
    
    # Align 1d indicators to 1d timeframe (no alignment needed for same timeframe)
    donch_upper_1d_aligned = donch_upper_1d
    donch_lower_1d_aligned = donch_lower_1d
    ema10_1d_aligned = ema10_1d
    atr_1d_aligned = atr_1d
    
    # Align 1w EMA50 to 1d timeframe
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    entry_price = 0.0
    atr_multiplier = 2.5  # ATR stoploss multiplier
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donch_upper_1d_aligned[i]) or 
            np.isnan(donch_lower_1d_aligned[i]) or 
            np.isnan(ema10_1d_aligned[i]) or 
            np.isnan(atr_1d_aligned[i]) or 
            np.isnan(ema50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        upper = donch_upper_1d_aligned[i]
        lower = donch_lower_1d_aligned[i]
        ema10 = ema10_1d_aligned[i]
        atr = atr_1d_aligned[i]
        ema50 = ema50_1w_aligned[i]
        
        # Determine 1w trend: price above/below 50 EMA
        uptrend = price > ema50
        downtrend = price < ema50
        
        if position == 0:
            # Long: price breaks above Donchian upper with 1w uptrend
            if price > upper and uptrend:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price breaks below Donchian lower with 1w downtrend
            elif price < lower and downtrend:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Exit conditions for long
            exit_signal = False
            # Exit if price reverts to EMA10
            if price <= ema10:
                exit_signal = True
            # Exit if ATR stoploss hit (using close price)
            elif price < entry_price - (atr_multiplier * atr):
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit conditions for short
            exit_signal = False
            # Exit if price reverts to EMA10
            if price >= ema10:
                exit_signal = True
            # Exit if ATR stoploss hit (using close price)
            elif price > entry_price + (atr_multiplier * atr):
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_1wEMA50_EMA10_ATRStop"
timeframe = "1d"
leverage = 1.0