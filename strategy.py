#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and ATR-based position sizing.
Long when price breaks above 20-day high AND weekly EMA50 is rising (bullish regime).
Short when price breaks below 20-day low AND weekly EMA50 is falling (bearish regime).
Exit when price reverts to 10-day EMA or ATR stoploss is hit.
Uses discrete position sizes (0.0, ±0.25) to minimize fee churn. Designed to work in both bull and bear markets via regime filter.
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
    
    # Get 1d data for Donchian channels and EMA
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
        lower = np.full_like(low, np.nan)
        for i in range(period-1, len(high)):
            upper[i] = np.max(high[i-period+1:i+1])
            lower[i] = np.min(low[i-period+1:i+1])
        return upper, lower
    
    donch_upper_1d, donch_lower_1d = donchian_channels(high_1d, low_1d, 20)
    
    # Calculate 1d EMA10 for exit
    def ema(values, period):
        return pd.Series(values).ewm(span=period, adjust=False, min_periods=period).mean().values
    
    ema10_1d = ema(close_1d, 10)
    
    # Calculate 1w EMA50 for trend filter
    ema50_1w = ema(close_1w, 50)
    ema50_1w_rising = np.zeros_like(close_1w, dtype=bool)
    ema50_1w_falling = np.zeros_like(close_1w, dtype=bool)
    for i in range(1, len(ema50_1w)):
        ema50_1w_rising[i] = ema50_1w[i] > ema50_1w[i-1]
        ema50_1w_falling[i] = ema50_1w[i] < ema50_1w[i-1]
    
    # Calculate ATR (14) for stoploss and position sizing
    def atr(high, low, close, period=14):
        tr = np.zeros_like(close)
        for i in range(1, len(close)):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        atr_val = np.zeros_like(close)
        atr_val[period] = np.mean(tr[1:period+1])
        for i in range(period+1, len(tr)):
            atr_val[i] = (atr_val[i-1] * (period-1) + tr[i]) / period
        return atr_val
    
    atr_1d = atr(high_1d, low_1d, close_1d, 14)
    
    # Align 1d indicators to 1d timeframe (no alignment needed as we're already on 1d)
    donch_upper_1d_aligned = donch_upper_1d
    donch_lower_1d_aligned = donch_lower_1d
    ema10_1d_aligned = ema10_1d
    atr_1d_aligned = atr_1d
    
    # Align 1w indicators to 1d timeframe
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    ema50_1w_rising_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w_rising.astype(float))
    ema50_1w_falling_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w_falling.astype(float))
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    entry_price = 0.0
    atr_multiplier = 2.5
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donch_upper_1d_aligned[i]) or 
            np.isnan(donch_lower_1d_aligned[i]) or 
            np.isnan(ema10_1d_aligned[i]) or 
            np.isnan(atr_1d_aligned[i]) or
            np.isnan(ema50_1w_aligned[i]) or
            np.isnan(ema50_1w_rising_aligned[i]) or
            np.isnan(ema50_1w_falling_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        donch_upper = donch_upper_1d_aligned[i]
        donch_lower = donch_lower_1d_aligned[i]
        ema10 = ema10_1d_aligned[i]
        atr_val = atr_1d_aligned[i]
        ema50_rising = bool(ema50_1w_rising_aligned[i])
        ema50_falling = bool(ema50_1w_falling_aligned[i])
        
        if position == 0:
            # Long: price breaks above Donchian upper AND weekly EMA50 is rising
            if price > donch_upper and ema50_rising:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price breaks below Donchian lower AND weekly EMA50 is falling
            elif price < donch_lower and ema50_falling:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Exit long: price reverts to EMA10 OR ATR stoploss hit
            if price <= ema10 or price <= entry_price - atr_multiplier * atr_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price reverts to EMA10 OR ATR stoploss hit
            if price >= ema10 or price >= entry_price + atr_multiplier * atr_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_1wEMA50_EMA10_ATRStop"
timeframe = "1d"
leverage = 1.0