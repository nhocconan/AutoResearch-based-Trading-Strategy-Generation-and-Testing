#!/usr/bin/env python3
"""
6h Williams Alligator + Elder Ray momentum with 1d trend filter
Hypothesis: Williams Alligator (jaw/teeth/lips) identifies trend phases, Elder Ray (bull/bear power) measures momentum strength.
Combined with 1d EMA50 trend filter to avoid counter-trend trades. Works in bull (buy when bull power > 0 and price above teeth in uptrend) 
and bear (sell when bear power > 0 and price below teeth in downtrend). Uses 6h for reduced noise vs lower timeframes.
Target: 50-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_alligator_elder_ray_1d_trend"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # 13-period ATR for stoploss
    atr = np.full(n, np.nan)
    if n >= 14:
        tr = np.maximum(
            high[1:] - low[1:],
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        )
        if len(tr) > 0:
            atr[1] = tr[0]
            for i in range(2, n):
                atr[i] = (tr[i-1] * 12 + atr[i-1]) / 13
    
    # Williams Alligator (13,8,5 SMMA) - equivalent to SMMA with period
    # SMMA is similar to EMA but with alpha = 1/period
    def smma(arr, period):
        res = np.full_like(arr, np.nan)
        if len(arr) >= period:
            res[period-1] = np.mean(arr[:period])
            for i in range(period, len(arr)):
                res[i] = (arr[i] + res[i-1] * (period-1)) / period
        return res
    
    # Alligator lines: jaw(13), teeth(8), lips(5)
    jaw = smma(close, 13)
    teeth = smma(close, 8)
    lips = smma(close, 5)
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = EMA13 - Low
    def ema(arr, period):
        res = np.full_like(arr, np.nan)
        if len(arr) >= period:
            multiplier = 2 / (period + 1)
            res[period-1] = np.mean(arr[:period])
            for i in range(period, len(arr)):
                res[i] = (arr[i] * multiplier) + (res[i-1] * (1 - multiplier))
        return res
    
    ema13 = ema(close, 13)
    bull_power = high - ema13
    bear_power = ema13 - low
    
    # Get 1d data for trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # EMA50 on 1d close
    ema_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 50:
        ema_1d[49] = np.mean(close_1d[:50])
        for i in range(50, len(close_1d)):
            ema_1d[i] = (close_1d[i] * 2 + ema_1d[i-1] * 48) / 50
    
    # 1d trend: above EMA50 = bullish, below = bearish
    trend_1d = np.where(close_1d > ema_1d, 1, -1)
    
    # Align 1d trend to 6h timeframe
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = 30  # Need enough data for Alligator and EMA
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(atr[i]) or np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or np.isnan(trend_1d_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            bars_since_entry += 1
            continue
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price crosses below lips OR against 1d trend OR stoploss
            if (close[i] < lips[i] or
                trend_1d_aligned[i] == -1 or
                close[i] < entry_price - 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.25
            bars_since_entry += 1
        elif position == -1:  # short position
            # Exit: price crosses above lips OR against 1d trend OR stoploss
            if (close[i] > lips[i] or
                trend_1d_aligned[i] == 1 or
                close[i] > entry_price + 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -0.25
            bars_since_entry += 1
        else:
            # Look for entries
            # Minimum holding period: only allow new entry after 4 bars flat
            if bars_since_entry >= 4:
                # Long: bull power > 0 (bullish momentum) AND price above teeth (trend alignment) AND 1d uptrend
                long_entry = (bull_power[i] > 0 and 
                             close[i] > teeth[i] and 
                             trend_1d_aligned[i] == 1)
                
                # Short: bear power > 0 (bearish momentum) AND price below teeth (trend alignment) AND 1d downtrend
                short_entry = (bear_power[i] > 0 and 
                              close[i] < teeth[i] and 
                              trend_1d_aligned[i] == -1)
                
                if long_entry:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                elif short_entry:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
                    bars_since_entry = 0
                else:
                    signals[i] = 0.0
                    bars_since_entry += 1
            else:
                signals[i] = 0.0
                bars_since_entry += 1
    
    return signals