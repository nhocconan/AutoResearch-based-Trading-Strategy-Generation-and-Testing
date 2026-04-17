#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and ATR-based stop.
Long when price breaks above 20-day high with 1w EMA50 uptrend.
Short when price breaks below 20-day low with 1w EMA50 downtrend.
Exit on opposite Donchian break or ATR trailing stop (2.5x ATR).
Uses discrete position sizing (0.25) to limit fee drag. Target: 30-80 trades over 4 years.
Works in bull via breakouts, in bear via short breakdowns with trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate 1d Donchian channels (20-period)
    def donchian_channels(high, low, period=20):
        upper = np.full_like(high, np.nan)
        lower = np.full_like(high, np.nan)
        for i in range(period-1, len(high)):
            upper[i] = np.max(high[i-period+1:i+1])
            lower[i] = np.min(low[i-period+1:i+1])
        return upper, lower
    
    upper_1d, lower_1d = donchian_channels(high_1d, low_1d, 20)
    
    # Calculate 1w EMA50
    def calculate_ema(values, period):
        ema = np.full_like(values, np.nan)
        if len(values) < period:
            return ema
        multiplier = 2 / (period + 1)
        ema[period-1] = np.mean(values[:period])
        for i in range(period, len(values)):
            ema[i] = (values[i] - ema[i-1]) * multiplier + ema[i-1]
        return ema
    
    ema_50_1w = calculate_ema(close_1w, 50)
    
    # Calculate ATR for stoploss (using 1d data)
    def calculate_atr(high, low, close, period=14):
        tr = np.zeros_like(high)
        for i in range(1, len(high)):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        atr = np.zeros_like(high)
        atr[period] = np.mean(tr[1:period+1])
        for i in range(period+1, len(tr)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        return atr
    
    atr_1d = calculate_atr(high_1d, low_1d, close_1d, 14)
    
    # Align indicators to primary timeframe (1d)
    upper_1d_aligned = align_htf_to_ltf(prices, df_1d, upper_1d)
    lower_1d_aligned = align_htf_to_ltf(prices, df_1d, lower_1d)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    start_idx = 60  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper_1d_aligned[i]) or 
            np.isnan(lower_1d_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(atr_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        upper = upper_1d_aligned[i]
        lower = lower_1d_aligned[i]
        ema_trend = ema_50_1w_aligned[i]
        atr = atr_1d_aligned[i]
        
        if position == 0:
            # Long: price breaks above upper Donchian with 1w EMA50 uptrend
            if price > upper and ema_trend > 0:
                signals[i] = 0.25
                position = 1
                entry_price = price
                highest_since_entry = price
            # Short: price breaks below lower Donchian with 1w EMA50 downtrend
            elif price < lower and ema_trend < 0:
                signals[i] = -0.25
                position = -1
                entry_price = price
                lowest_since_entry = price
        
        elif position == 1:
            # Update highest price since entry
            highest_since_entry = max(highest_since_entry, price)
            
            # Exit conditions:
            # 1. Price breaks below lower Donchian (contrarian signal)
            # 2. ATR trailing stop: price drops 2.5*ATR from highest since entry
            if price < lower or price < highest_since_entry - 2.5 * atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Update lowest price since entry
            lowest_since_entry = min(lowest_since_entry, price)
            
            # Exit conditions:
            # 1. Price breaks above upper Donchian (contrarian signal)
            # 2. ATR trailing stop: price rises 2.5*ATR from lowest since entry
            if price > upper or price > lowest_since_entry + 2.5 * atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_1wEMA50_ATRTrail"
timeframe = "1d"
leverage = 1.0