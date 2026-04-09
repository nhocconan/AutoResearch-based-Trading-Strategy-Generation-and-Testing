#!/usr/bin/env python3
# 12h_daily_camarilla_pivot_volume_chop_v1
# Hypothesis: 12h strategy using daily Camarilla pivot levels (S3/S4 for shorts, R3/R4 for longs) 
# with volume confirmation (>1.3x 20-period average) and choppiness regime filter (CHOP > 61.8 for mean reversion).
# Daily pivots act as strong support/resistance; mean reversion in choppy markets captures reversals at extremes.
# Uses discrete sizing (0.0, ±0.25) to minimize fee churn. Target: 12-25 trades/year.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_daily_camarilla_pivot_volume_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d HTF data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_d = df_1d['high'].values
    low_d = df_1d['low'].values
    close_d = df_1d['close'].values
    
    # Daily Camarilla pivot levels (based on previous day)
    # Pivot = (high + low + close) / 3
    pivot = (high_d + low_d + close_d) / 3.0
    # Range = high - low
    rng = high_d - low_d
    # Resistance levels
    r4 = close_d + rng * 1.500
    r3 = close_d + rng * 1.250
    r2 = close_d + rng * 1.166
    r1 = close_d + rng * 1.083
    # Support levels
    s1 = close_d - rng * 1.083
    s2 = close_d - rng * 1.166
    s3 = close_d - rng * 1.250
    s4 = close_d - rng * 1.500
    
    # Align daily Camarilla data to 12h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Choppiness Index (14-period) for regime filter
    def true_range(h, l, c_prev):
        tr1 = h - l
        tr2 = np.abs(h - c_prev)
        tr3 = np.abs(l - c_prev)
        return np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Calculate true range
    close_shift = np.roll(close, 1)
    close_shift[0] = close[0]
    tr = true_range(high, low, close_shift)
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate directional movement
    dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                       np.maximum(high - np.roll(high, 1), 0), 0)
    dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    # Handle first element
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smooth DM and TR
    dm_plus_smooth = pd.Series(dm_plus).rolling(window=14, min_periods=1).sum().values
    dm_minus_smooth = pd.Series(dm_minus).rolling(window=14, min_periods=1).sum().values
    tr_smooth = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Calculate DI+ and DI-
    di_plus = np.where(tr_smooth != 0, 100 * dm_plus_smooth / tr_smooth, 0)
    di_minus = np.where(tr_smooth != 0, 100 * dm_minus_smooth / tr_smooth, 0)
    
    # Calculate DX and CHOP
    dx = np.where((di_plus + di_minus) != 0, 
                  100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    chop = np.where(dx != 0, 
                    100 * np.log10(tr_smooth / np.sqrt(14)) / np.log10(dx), 50)
    # Handle edge cases
    chop = np.where(np.isnan(chop) | np.isinf(chop), 50, chop)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(r4_aligned[i]) or np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(volume_ma[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3x 20-period average
        volume_confirmed = volume[i] > 1.3 * volume_ma[i]
        
        # Choppiness regime: CHOP > 61.8 indicates ranging market (good for mean reversion)
        chop_regime = chop[i] > 61.8
        
        if position == 1:  # Long position
            # Exit: price falls below R3 OR chop regime ends
            if close[i] < r3_aligned[i] or not chop_regime:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price rises above S3 OR chop regime ends
            if close[i] > s3_aligned[i] or not chop_regime:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            if volume_confirmed and chop_regime:
                # Long entry: price above R4 AND choppy market (mean reversion from extreme)
                if close[i] > r4_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short entry: price below S4 AND choppy market (mean reversion from extreme)
                elif close[i] < s4_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals