#!/usr/bin/env python3
"""
12h_Camarilla_Pivot_Breakout_Volume_Trend
Hypothesis: Camarilla pivot levels from daily timeframe identify key support/resistance zones.
Breakouts above/below these levels with volume confirmation and weekly ADX trend filter capture
institutional moves. Works in bull markets by catching breakouts and in bear markets by avoiding
false signals via trend filter. Targets low trade frequency (12-37/year) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Load 1w data ONCE before loop for ADX trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # === Calculate Camarilla pivot levels from previous day ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: R4, R3, R2, R1, PP, S1, S2, S3, S4
    # Based on previous day's OHLC
    camarilla_r4 = np.zeros_like(close_1d)
    camarilla_r3 = np.zeros_like(close_1d)
    camarilla_r2 = np.zeros_like(close_1d)
    camarilla_r1 = np.zeros_like(close_1d)
    camarilla_pp = np.zeros_like(close_1d)
    camarilla_s1 = np.zeros_like(close_1d)
    camarilla_s2 = np.zeros_like(close_1d)
    camarilla_s3 = np.zeros_like(close_1d)
    camarilla_s4 = np.zeros_like(close_1d)
    
    for i in range(1, len(close_1d)):
        # Use previous day's data to calculate today's levels
        h = high_1d[i-1]
        l = low_1d[i-1]
        c = close_1d[i-1]
        rng = h - l
        
        camarilla_pp[i] = (h + l + c) / 3
        camarilla_r1[i] = c + (rng * 1.1 / 12)
        camarilla_r2[i] = c + (rng * 1.1 / 6)
        camarilla_r3[i] = c + (rng * 1.1 / 4)
        camarilla_r4[i] = c + (rng * 1.1 / 2)
        camarilla_s1[i] = c - (rng * 1.1 / 12)
        camarilla_s2[i] = c - (rng * 1.1 / 6)
        camarilla_s3[i] = c - (rng * 1.1 / 4)
        camarilla_s4[i] = c - (rng * 1.1 / 2)
    
    # For first day, use same day's data (will not trigger until second day anyway)
    camarilla_r4[0] = camarilla_r4[1] if len(camarilla_r4) > 1 else 0
    camarilla_r3[0] = camarilla_r3[1] if len(camarilla_r3) > 1 else 0
    camarilla_r2[0] = camarilla_r2[1] if len(camarilla_r2) > 1 else 0
    camarilla_r1[0] = camarilla_r1[1] if len(camarilla_r1) > 1 else 0
    camarilla_pp[0] = camarilla_pp[1] if len(camarilla_pp) > 1 else 0
    camarilla_s1[0] = camarilla_s1[1] if len(camarilla_s1) > 1 else 0
    camarilla_s2[0] = camarilla_s2[1] if len(camarilla_s2) > 1 else 0
    camarilla_s3[0] = camarilla_s3[1] if len(camarilla_s3) > 1 else 0
    camarilla_s4[0] = camarilla_s4[1] if len(camarilla_s4) > 1 else 0
    
    # Align Camarilla levels to 12h timeframe
    camarilla_r1_12h = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_r2_12h = align_htf_to_ltf(prices, df_1d, camarilla_r2)
    camarilla_r3_12h = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s1_12h = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    camarilla_s2_12h = align_htf_to_ltf(prices, df_1d, camarilla_s2)
    camarilla_s3_12h = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # === Weekly ADX(14) for trend filter ===
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate directional movement
    high_diff = np.diff(high_1w)
    low_diff = -np.diff(low_1w)
    
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
        if period == 1:
            result[0] = arr[0]
            for i in range(1, len(arr)):
                result[i] = (result[i-1] * 0 + arr[i]) / 1
        else:
            result[period-1] = np.nanmean(arr[1:period]) if np.any(~np.isnan(arr[1:period])) else arr[0]
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
    
    # === 12h Volume confirmation ===
    vol_ma = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio = prices['volume'].values / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after volume MA warmup
        # Skip if indicators not ready
        if (np.isnan(camarilla_r1_12h[i]) or np.isnan(camarilla_r2_12h[i]) or
            np.isnan(camarilla_r3_12h[i]) or np.isnan(camarilla_s1_12h[i]) or
            np.isnan(camarilla_s2_12h[i]) or np.isnan(camarilla_s3_12h[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(adx_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        price_open = prices['open'].iloc[i]
        adx_val = adx_1w_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        if position == 0:
            # Long: price breaks above Camarilla R3 with volume and trend
            if (price_close > camarilla_r3_12h[i] and
                vol_ratio_val > 1.5 and
                adx_val > 20):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Camarilla S3 with volume and trend
            elif (price_close < camarilla_s3_12h[i] and
                  vol_ratio_val > 1.5 and
                  adx_val > 20):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit when price returns to opposite Camarilla level
            if position == 1 and price_close < camarilla_s3_12h[i]:
                signals[i] = 0.0
                position = 0
            elif position == -1 and price_close > camarilla_r3_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_Camarilla_Pivot_Breakout_Volume_Trend"
timeframe = "12h"
leverage = 1.0