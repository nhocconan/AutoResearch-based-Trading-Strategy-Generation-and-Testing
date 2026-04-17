#!/usr/bin/env python3
"""
4h_1D_Pivot_R1_S1_Breakout_Volume_Trend
Hypothesis: On 4h timeframe, buy when price breaks above 1D Camarilla R1 with volume spike (>1.5x median volume) and trend alignment (price > 4h EMA50), sell when breaks below 1D S1. Uses volume and trend filters to avoid false breakouts. Target: 20-30 trades/year for low fee drag. Works in bull/bear via trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_atr(high, low, close, period=14):
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    def smooth_wilder(arr, period):
        result = np.full_like(arr, np.nan)
        if len(arr) < period:
            return result
        result[period-1] = np.nansum(arr[1:period])
        for i in range(period, len(arr)):
            if not np.isnan(arr[i]) and not np.isnan(result[i-1]):
                result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    return smooth_wilder(tr, period)

def calculate_camarilla(high, low, close):
    range_hl = high - low
    r1 = close + range_hl * 1.1 / 12
    s1 = close - range_hl * 1.1 / 12
    return r1, s1

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === 1D Data (HTF for Camarilla levels, volume) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # 1D ATR (14-period) for trend filter (optional)
    atr_1d = calculate_atr(high_1d, low_1d, close_1d, 14)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # 1D median ATR (50-period) for expansion filter
    atr_median_1d = pd.Series(atr_1d).rolling(window=50, min_periods=50).median().values
    atr_median_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_median_1d)
    
    # 1D Camarilla levels (R1, S1)
    r1_1d, s1_1d = calculate_camarilla(high_1d, low_1d, close_1d)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # 1D median volume (50-period) for volume spike filter
    vol_median_1d = pd.Series(volume_1d).rolling(window=50, min_periods=50).median().values
    vol_median_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_median_1d)
    
    # 4h EMA50 for trend filter
    close_series = pd.Series(close)
    ema50 = close_series.ewm(span=50, adjust=False, min_periods=50).values
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(atr_1d_aligned[i]) or 
            np.isnan(atr_median_1d_aligned[i]) or
            np.isnan(r1_1d_aligned[i]) or
            np.isnan(s1_1d_aligned[i]) or
            np.isnan(vol_median_1d_aligned[i]) or
            np.isnan(ema50[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Get current 1D bar's volume for confirmation
        vol_1d_current = align_htf_to_ltf(prices, df_1d, volume_1d)[i]
        
        # Volume spike: current volume > 1.5x median volume
        vol_spike = vol_1d_current > 1.5 * vol_median_1d_aligned[i]
        
        # Trend filter: price > 4h EMA50 for long, price < 4h EMA50 for short
        trend_up = close[i] > ema50[i]
        trend_down = close[i] < ema50[i]
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: price breaks above 1D R1 with volume spike and uptrend
            if close[i] > r1_1d_aligned[i] and vol_spike and trend_up:
                signals[i] = 0.25
                position = 1
                continue
            # Short: price breaks below 1D S1 with volume spike and downtrend
            elif close[i] < s1_1d_aligned[i] and vol_spike and trend_down:
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit when price breaks below S1 (opposite breakout)
            if close[i] < s1_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit when price breaks above R1 (opposite breakout)
            if close[i] > r1_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_1D_Pivot_R1_S1_Breakout_Volume_Trend"
timeframe = "4h"
leverage = 1.0