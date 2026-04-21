#!/usr/bin/env python3
"""
4h_SMA_Crossover_Trend_Filter
Hypothesis: SMA crossovers capture established trends, volume confirms momentum, and 1-day ADX filters for trending regimes.
Works in bull markets by riding trends and in bear markets by avoiding false signals via ADX filter. Targets low trade frequency (20-50/year) to minimize fee drag.
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
    
    # === 1-day ADX(14) for trend filter ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate directional movement
    high_diff = np.diff(high_1d)
    low_diff = -np.diff(low_1d)
    
    plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
    minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
    
    # True range
    tr1 = np.abs(np.diff(high_1d))
    tr2 = np.abs(np.diff(low_1d))
    tr3 = np.abs(np.diff(close_1d))
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
        result[period-1] = np.nanmean(arr[1:period]) if period > 1 else arr[0]
        for i in range(period, len(arr)):
            if np.isnan(result[i-1]) or np.isnan(arr[i]):
                result[i] = np.nan
            else:
                result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    period = 14
    atr_1d = wilder_smooth(tr, period)
    plus_di = 100 * wilder_smooth(plus_dm, period) / atr_1d
    minus_di = 100 * wilder_smooth(minus_dm, period) / atr_1d
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx_1d = wilder_smooth(dx, period)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # === 4h SMA(50) and SMA(200) crossover ===
    close = prices['close'].values
    sma_50 = pd.Series(close).rolling(window=50, min_periods=50).mean().values
    sma_200 = pd.Series(close).rolling(window=200, min_periods=200).mean().values
    
    # === 4h Volume confirmation ===
    vol_ma = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio = prices['volume'].values / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):  # Start after SMA200 warmup
        # Skip if indicators not ready
        if (np.isnan(sma_50[i]) or np.isnan(sma_200[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(adx_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        adx_val = adx_1d_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        if position == 0:
            # Long: SMA50 crosses above SMA200 with volume and trend
            if (sma_50[i] > sma_200[i] and
                vol_ratio_val > 1.5 and
                adx_val > 25):
                signals[i] = 0.25
                position = 1
            # Short: SMA50 crosses below SMA200 with volume and trend
            elif (sma_50[i] < sma_200[i] and
                  vol_ratio_val > 1.5 and
                  adx_val > 25):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit when SMA50 crosses back in opposite direction
            if position == 1 and sma_50[i] < sma_200[i]:
                signals[i] = 0.0
                position = 0
            elif position == -1 and sma_50[i] > sma_200[i]:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_SMA_Crossover_Trend_Filter"
timeframe = "4h"
leverage = 1.0