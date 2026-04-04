#!/usr/bin/env python3
"""
Experiment #2707: 6h Camarilla pivot levels from 1d + volume confirmation + ADX regime filter
HYPOTHESIS: Camarilla pivot levels (R3/S3 for mean reversion, R4/S4 for breakout) from daily timeframe
provide institutional support/resistance. Combined with volume spike and ADX>25 for trending markets,
this captures both mean reversion in range and breakout continuation in trends. 6h timeframe avoids
overtrading while allowing precise entry. Target: 75-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_2707_6h_camarilla_1d_vol_adx_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for Camarilla pivots and ADX (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels from previous day
    # Pivot = (H + L + C) / 3
    # Range = H - L
    # R3 = Pivot + Range * 1.1/2
    # S3 = Pivot - Range * 1.1/2
    # R4 = Pivot + Range * 1.1
    # S4 = Pivot - Range * 1.1
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    r3_1d = pivot_1d + range_1d * 1.1 / 2.0
    s3_1d = pivot_1d - range_1d * 1.1 / 2.0
    r4_1d = pivot_1d + range_1d * 1.1
    s4_1d = pivot_1d - range_1d * 1.1
    
    # Align Camarilla levels to 6h timeframe (shifted by 1 for completed day only)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # Calculate ADX(14) on 1d for regime filter
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])  # align indices
        
        # Directional Movement
        dm_plus = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), 
                           np.maximum(high[1:] - high[:-1], 0), 0)
        dm_plus = np.concatenate([[np.nan], dm_plus])
        dm_minus = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), 
                            np.maximum(low[:-1] - low[1:], 0), 0)
        dm_minus = np.concatenate([[np.nan], dm_minus])
        
        # Smoothed values
        atr = pd.Series(tr).ewm(alpha=1/period, adjust=False).mean().values
        dm_plus_smooth = pd.Series(dm_plus).ewm(alpha=1/period, adjust=False).mean().values
        dm_minus_smooth = pd.Series(dm_minus).ewm(alpha=1/period, adjust=False).mean().values
        
        # Directional Indicators
        di_plus = 100 * dm_plus_smooth / (atr + 1e-10)
        di_minus = 100 * dm_minus_smooth / (atr + 1e-10)
        
        # ADX
        dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
        adx = pd.Series(dx).ewm(alpha=1/period, adjust=False).mean().values
        return adx
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # === 6h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    warmup = 50  # sufficient for all indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(r3_1d_aligned[i]) or np.isnan(s3_1d_aligned[i]) or
            np.isnan(r4_1d_aligned[i]) or np.isnan(s4_1d_aligned[i]) or
            np.isnan(adx_1d_aligned[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Mean Reversion at R3/S3 (price reverts to pivot) ---
        # Long: price touches/slightly breaks S3 but closes back above it (rejection)
        # Short: price touches/slightly breaks R3 but closes back below it (rejection)
        mean_rev_long = (low[i] <= s3_1d_aligned[i] * 1.002 and close[i] > s3_1d_aligned[i])
        mean_rev_short = (high[i] >= r3_1d_aligned[i] * 0.998 and close[i] < r3_1d_aligned[i])
        
        # --- Breakout Continuation at R4/S4 (strong momentum) ---
        # Long: price breaks above R4 with conviction
        # Short: price breaks below S4 with conviction
        breakout_long = (close[i] > r4_1d_aligned[i] and high[i] > r4_1d_aligned[i] * 1.001)
        breakout_short = (close[i] < s4_1d_aligned[i] and low[i] < s4_1d_aligned[i] * 0.999)
        
        # --- Filters ---
        # Volume confirmation: require volume spike (> 1.8x average)
        volume_spike = vol_ratio[i] > 1.8
        
        # ADX regime filter: only trade when trending (ADX > 25) OR at extremes in range (ADX < 20)
        # Actually, we want to fade at R3/S3 in ranging markets (ADX < 25) and breakout at R4/S4 in trending (ADX > 25)
        adx_val = adx_1d_aligned[i]
        fading_regime = adx_val < 25  # ranging/choppy - good for mean reversion
        trending_regime = adx_val >= 25  # trending - good for breakouts
        
        # --- Entry Logic ---
        if volume_spike:
            # Mean reversion entries in ranging market
            if fading_regime:
                if mean_rev_long:
                    signals[i] = SIZE
                elif mean_rev_short:
                    signals[i] = -SIZE
            # Breakout continuation entries in trending market
            elif trending_regime:
                if breakout_long:
                    signals[i] = SIZE
                elif breakout_short:
                    signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals