#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla Pivot Reversal with 1d Volume Spike and ADX Trend Filter
# Uses daily Camarilla pivot levels (S3/S4 for longs, R3/R4 for shorts) as key support/resistance.
# Requires volume > 2x 20-bar median to confirm institutional interest at pivot levels.
# ADX > 25 ensures we only trade in trending markets, avoiding false reversals in chop.
# Designed for 12h timeframe to capture medium-term reversals with low trade frequency.
# Works in bull markets (buy S3/S4 bounces) and bear markets (sell R3/R4 rejections).
# Conservative sizing (0.25) to limit trade frequency and avoid fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily Camarilla pivot levels (based on previous day's OHLC)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot and support/resistance levels
    pivot = (high_1d + low_1d + close_1d) / 3.0
    range_hl = high_1d - low_1d
    
    # Camarilla levels
    S1 = close_1d - (range_hl * 1.0 / 6.0)
    S2 = close_1d - (range_hl * 2.0 / 6.0)
    S3 = close_1d - (range_hl * 3.0 / 6.0)
    S4 = close_1d - (range_hl * 4.0 / 6.0)
    R1 = close_1d + (range_hl * 1.0 / 6.0)
    R2 = close_1d + (range_hl * 2.0 / 6.0)
    R3 = close_1d + (range_hl * 3.0 / 6.0)
    R4 = close_1d + (range_hl * 4.0 / 6.0)
    
    # Align to 12h timeframe (values from previous day's close)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    S4_aligned = align_htf_to_ltf(prices, df_1d, S4)
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    R4_aligned = align_htf_to_ltf(prices, df_1d, R4)
    
    # 14-period ADX for trend strength (daily)
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values
    tr14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_plus_14 = pd.Series(dm_plus).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_minus_14 = pd.Series(dm_minus).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # DI and DX
    di_plus = 100 * dm_plus_14 / tr14
    di_minus = 100 * dm_minus_14 / tr14
    dx = np.where((di_plus + di_minus) > 0, 
                  100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    
    # ADX
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume confirmation: current > 2x median of last 20 bars
    vol_median = pd.Series(volume).rolling(window=20, min_periods=1).median()
    vol_threshold = 2.0 * vol_median
    
    signals = np.zeros(n)
    
    for i in range(20, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(pivot_aligned[i]) or np.isnan(S3_aligned[i]) or np.isnan(S4_aligned[i]) or
            np.isnan(R3_aligned[i]) or np.isnan(R4_aligned[i]) or np.isnan(adx_aligned[i]) or
            np.isnan(vol_threshold[i])):
            continue
        
        # Long: Price at S3/S4 with volume spike and ADX > 25
        if ((close[i] <= S3_aligned[i] * 1.005 or close[i] <= S4_aligned[i] * 1.005) and
            volume[i] > vol_threshold[i] and adx_aligned[i] > 25):
            signals[i] = 0.25
        
        # Short: Price at R3/R4 with volume spike and ADX > 25
        elif ((close[i] >= R3_aligned[i] * 0.995 or close[i] >= R4_aligned[i] * 0.995) and
              volume[i] > vol_threshold[i] and adx_aligned[i] > 25):
            signals[i] = -0.25
        
        # Exit: price moves back toward pivot (mean reversion completion)
        elif (i > 0 and 
              ((signals[i-1] == 0.25 and close[i] >= pivot_aligned[i] * 0.995) or
               (signals[i-1] == -0.25 and close[i] <= pivot_aligned[i] * 1.005))):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "12h_CamarillaPivot_Volume_ADXFilter"
timeframe = "12h"
leverage = 1.0