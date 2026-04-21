#!/usr/bin/env python3
"""
1d_KAMA_Reverse_Cross_With_Volume_Filter
Hypothesis: Kaufman Adaptive Moving Average (KAMA) adapts to market noise, providing reliable trend signals.
Go long when KAMA turns up from below price with volume confirmation; short when KAMA turns down from above price.
Use 1-week ADX > 20 to filter for trending markets only. Designed for low trade frequency (7-25 trades/year) to minimize fee drag.
Works in bull markets by catching trends and in bear markets by avoiding whipsaws via ADX filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === 1-week ADX(14) for trend filter ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate directional movement
    high_diff = np.diff(high_1w)
    low_diff = -np.diff(low_1w)  # inverted for calculation
    
    plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
    minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
    
    # True range
    tr1 = np.abs(np.diff(high_1w))
    tr2 = np.abs(np.diff(low_1w))
    tr3 = np.abs(np.diff(close_1w))
    tr = np.maximum.reduce([tr1, tr2, tr3])
    
    # Add first element (no diff)
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    tr = np.concatenate([[0], tr])
    
    # Wilder's smoothing
    def wilder_smooth(arr, period):
        result = np.full_like(arr, np.nan)
        if len(arr) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(arr[1:period]) if period > 1 else arr[0]
        # Wilder smoothing: today = (yesterday * (period-1) + today) / period
        for i in range(period, len(arr)):
            if np.isnan(result[i-1]) or np.isnan(arr[i]):
                result[i] = np.nan
            else:
                result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    period = 14
    atr_1w = wilder_smooth(tr, period)
    plus_di = 100 * wilder_smooth(plus_dm, period) / atr_1w
    minus_di = 100 * wilder_smooth(minus_dm, period) / atr_1w
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx_1w = wilder_smooth(dx, period)
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    # === 1d KAMA(10,2,30) ===
    close_1d = df_1d['close'].values
    # Efficiency Ratio
    change = np.abs(np.diff(close_1d))
    abs_change = np.abs(np.diff(close_1d))
    direction = np.abs(np.diff(close_1d, 10))  # 10-period net change
    volatility = np.nansum(abs_change.reshape(-1, 10), axis=1)  # 10-period sum of absolute changes
    # Pad arrays
    change = np.concatenate([[0], change])
    direction = np.concatenate([np.zeros(9), direction])
    volatility = np.concatenate([np.zeros(9), volatility])
    er = np.where(volatility != 0, direction / volatility, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    # KAMA calculation
    kama = np.full_like(close_1d, np.nan)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        if np.isnan(sc[i]) or np.isnan(kama[i-1]):
            kama[i] = np.nan
        else:
            kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    # === 1d Volume spike detection ===
    vol_ma = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio = df_1d['volume'].values / vol_ma
    vol_ratio_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # Start after warmup
        # Skip if indicators not ready
        if (np.isnan(kama_aligned[i]) or
            np.isnan(adx_1w_aligned[i]) or
            np.isnan(vol_ratio_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        kama_val = kama_aligned[i]
        adx_val = adx_1w_aligned[i]
        vol_ratio_val = vol_ratio_aligned[i]
        
        if position == 0:
            # Long: KAMA turns up (price crosses above KAMA from below) with volume confirmation and trending market
            if (price_close > kama_val and 
                prices['close'].iloc[i-1] <= kama_aligned[i-1] and  # was below or equal yesterday
                vol_ratio_val > 1.5 and 
                adx_val > 20):
                signals[i] = 0.25
                position = 1
            # Short: KAMA turns down (price crosses below KAMA from above) with volume confirmation and trending market
            elif (price_close < kama_val and 
                  prices['close'].iloc[i-1] >= kama_aligned[i-1] and  # was above or equal yesterday
                  vol_ratio_val > 1.5 and 
                  adx_val > 20):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit when price crosses KAMA in opposite direction
            if position == 1 and price_close < kama_val:
                signals[i] = 0.0
                position = 0
            elif position == -1 and price_close > kama_val:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_KAMA_Reverse_Cross_With_Volume_Filter"
timeframe = "1d"
leverage = 1.0