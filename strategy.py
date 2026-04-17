#!/usr/bin/env python3
"""
4h_Camarilla_Pivot_Breakout_Volume_Regime_v1
Camarilla pivot levels from 1d + volume spike + choppiness regime.
Breakouts above S3 or below R3 trigger entries in the direction of breakout.
Choppiness filter ensures we only trade in trending markets (CHOP < 38.2).
Designed to work in both bull and bear markets by using volatility-based pivot levels
and regime filtering to avoid chop.
Target: 20-50 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d OHLC for Camarilla pivot calculation ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels (R3, R2, R1, PP, S1, S2, S3)
    # Based on previous day's OHLC
    camarilla_r3 = np.zeros_like(close_1d)
    camarilla_r2 = np.zeros_like(close_1d)
    camarilla_r1 = np.zeros_like(close_1d)
    camarilla_pp = np.zeros_like(close_1d)
    camarilla_s1 = np.zeros_like(close_1d)
    camarilla_s2 = np.zeros_like(close_1d)
    camarilla_s3 = np.zeros_like(close_1d)
    
    for i in range(len(close_1d)):
        if i >= 1:  # Need previous day's data
            ph = high_1d[i-1]  # Previous high
            pl = low_1d[i-1]   # Previous low
            pc = close_1d[i-1] # Previous close
            
            camarilla_pp[i] = (ph + pl + pc) / 3
            camarilla_r1[i] = camarilla_pp[i] + (ph - pl) * 1.1 / 12
            camarilla_r2[i] = camarilla_pp[i] + (ph - pl) * 1.1 / 6
            camarilla_r3[i] = camarilla_pp[i] + (ph - pl) * 1.1 / 4
            camarilla_s1[i] = camarilla_pp[i] - (ph - pl) * 1.1 / 12
            camarilla_s2[i] = camarilla_pp[i] - (ph - pl) * 1.1 / 6
            camarilla_s3[i] = camarilla_pp[i] - (ph - pl) * 1.1 / 4
        else:
            # First bar: no previous data
            camarilla_pp[i] = camarilla_r1[i] = camarilla_r2[i] = camarilla_r3[i] = np.nan
            camarilla_s1[i] = camarilla_s2[i] = camarilla_s3[i] = np.nan
    
    # === 4h Choppiness Index (14-period) ===
    # Chop > 61.8 = ranging, Chop < 38.2 = trending
    tr14 = np.zeros(n)
    for i in range(n):
        if i >= 13:
            tr = np.maximum(
                high[i] - low[i],
                np.maximum(
                    np.abs(high[i] - close[i-1]),
                    np.abs(low[i] - close[i-1])
                )
            )
            # Sum of true range over 14 periods
            tr_sum = 0
            for j in range(14):
                idx = i - j
                if idx >= 0:
                    tr_j = np.maximum(
                        high[idx] - low[idx],
                        np.maximum(
                            np.abs(high[idx] - close[idx-1]) if idx > 0 else 0,
                            np.abs(low[idx] - close[idx-1]) if idx > 0 else 0
                        )
                    )
                    tr_sum += tr_j
            
            # Highest high and lowest low over 14 periods
            hh = np.max(high[i-13:i+1]) if i >= 13 else high[i]
            ll = np.min(low[i-13:i+1]) if i >= 13 else low[i]
            
            if tr_sum > 0 and hh > ll:
                chop = 100 * np.log10(tr_sum / (hh - ll)) / np.log10(14)
                tr14[i] = chop
            else:
                tr14[i] = 50  # neutral
        else:
            tr14[i] = 50
    
    # Chop < 38.2 = trending (favorable for breakouts)
    chop_filter = tr14 < 38.2
    
    # === 4h Volume confirmation (20-period average) ===
    vol_ma_20 = np.full(n, np.nan)
    for i in range(n):
        if i >= 19:
            vol_ma_20[i] = np.mean(volume[i-19:i+1])
        elif i > 0:
            vol_ma_20[i] = np.mean(volume[max(0, i-9):i+1])
        else:
            vol_ma_20[i] = volume[0]
    
    vol_confirm = volume > vol_ma_20 * 1.5  # volume spike: 1.5x average
    
    # === Align Camarilla levels from 1d to 4h timeframe ===
    r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    r2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r2)
    r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    pp_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pp)
    s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    s2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s2)
    s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(chop_filter[i]) or np.isnan(vol_confirm[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: price breaks above R3 with volume and in trending regime
            if (close[i] > r3_aligned[i] and 
                vol_confirm[i] and 
                chop_filter[i]):
                signals[i] = 0.25
                position = 1
                continue
            # Short: price breaks below S3 with volume and in trending regime
            elif (close[i] < s3_aligned[i] and 
                  vol_confirm[i] and 
                  chop_filter[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: price crosses below pivot point (PP) OR opposite breakout (below S3)
            if (close[i] < pp_aligned[i] or 
                close[i] < s3_aligned[i]):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above pivot point (PP) OR opposite breakout (above R3)
            if (close[i] > pp_aligned[i] or 
                close[i] > r3_aligned[i]):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_Pivot_Breakout_Volume_Regime_v1"
timeframe = "4h"
leverage = 1.0