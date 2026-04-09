#!/usr/bin/env python3
# 6h_adx_alligator_pivot_v1
# Hypothesis: 6h strategy using ADX trend strength + Williams Alligator (smoothed medians) for direction + 12h pivot levels for entry/exit.
# ADX > 25 filters for trending markets, Alligator jaw/teeth/lips alignment confirms trend strength.
# Pivot breakouts provide precise entries with trend confirmation. Designed for low-frequency, high-conviction trades.
# Works in bull/bear via ADX regime filter and pivot structure respecting institutional levels.
# Target: 50-150 total trades over 4 years (12-37/year). Size: 0.25.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_adx_alligator_pivot_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h HTF data for pivot points and trend context
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # 12h Pivot points (standard floor pivot)
    prev_close_12h = np.roll(close_12h, 1)
    prev_high_12h = np.roll(high_12h, 1)
    prev_low_12h = np.roll(low_12h, 1)
    prev_close_12h[0] = np.nan
    prev_high_12h[0] = np.nan
    prev_low_12h[0] = np.nan
    
    pivot_point = (prev_high_12h + prev_low_12h + prev_close_12h) / 3
    prev_range = prev_high_12h - prev_low_12h
    
    # Key pivot levels: R1, S1 (primary), R2, S2 (secondary)
    r1 = 2 * pivot_point - prev_low_12h
    s1 = 2 * pivot_point - prev_high_12h
    r2 = pivot_point + prev_range
    s2 = pivot_point - prev_range
    
    # Align pivot levels to 6h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_12h, r1)
    s1_aligned = align_htf_to_ltf(prices, df_12h, s1)
    r2_aligned = align_htf_to_ltf(prices, df_12h, r2)
    s2_aligned = align_htf_to_ltf(prices, df_12h, s2)
    
    # 6h Williams Alligator (smoothed medians)
    # Jaw: 13-period SMMA, Teeth: 8-period SMMA, Lips: 5-period SMMA
    def smma(arr, period):
        """Smoothed Moving Average"""
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        result = np.full_like(arr, np.nan, dtype=float)
        # First value is SMA
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (Prev SMMA*(period-1) + Current Price) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(close, 13)
    teeth = smma(close, 8)
    lips = smma(close, 5)
    
    # 6h ADX for trend strength (14-period)
    def calculate_adx(high, low, close, period=14):
        """Average Directional Index"""
        if len(high) < period + 1:
            return np.full_like(high, np.nan)
        
        # True Range
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
        
        # Directional Movement
        dm_plus = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), 
                           np.maximum(high[1:] - high[:-1], 0), 0)
        dm_plus = np.concatenate([[0], dm_plus])
        dm_minus = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), 
                            np.maximum(low[:-1] - low[1:], 0), 0)
        dm_minus = np.concatenate([[0], dm_minus])
        
        # Smoothed TR, DM+
        atr = pd.Series(tr).ewm(span=period, adjust=False, min_periods=period).mean().values
        dm_plus_smooth = pd.Series(dm_plus).ewm(span=period, adjust=False, min_periods=period).mean().values
        dm_minus_smooth = pd.Series(dm_minus).ewm(span=period, adjust=False, min_periods=period).mean().values
        
        # Directional Indicators
        di_plus = 100 * dm_plus_smooth / atr
        di_minus = 100 * dm_minus_smooth / atr
        
        # DX and ADX
        dx = np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100
        adx = pd.Series(dx).ewm(span=period, adjust=False, min_periods=period).mean().values
        
        return adx
    
    adx = calculate_adx(high, low, close)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Warmup for Alligator and ADX
        # Skip if any required data is NaN
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(adx[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Alligator alignment: Lips > Teeth > Jaw = bullish, Lips < Teeth < Jaw = bearish
        alligator_bullish = lips[i] > teeth[i] > jaw[i]
        alligator_bearish = lips[i] < teeth[i] < jaw[i]
        
        # ADX trend strength filter
        strong_trend = adx[i] > 25
        
        if position == 1:  # Long position
            # Exit: Alligator turns bearish OR price breaks below S1
            if not alligator_bullish or close[i] < s1_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Alligator turns bullish OR price breaks above R1
            if not alligator_bearish or close[i] > r1_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Need strong trend and Alligator alignment
            if strong_trend:
                # Long: Alligator bullish + price breaks above R1 with momentum
                if alligator_bullish and close[i] > r1_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short: Alligator bearish + price breaks below S1 with momentum
                elif alligator_bearish and close[i] < s1_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals