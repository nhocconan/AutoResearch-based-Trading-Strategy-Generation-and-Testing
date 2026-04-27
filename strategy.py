#!/usr/bin/env python3
"""
4h_TRIX_VolumeSpike_ChopRegime_Trend
Hypothesis: Uses TRIX (15-period EMA of EMA of EMA of close) for momentum, 
volume confirmation (>2.0x 20-period average), and choppiness regime filter (CHOP(14) > 61.8 for ranging, < 38.2 for trending).
Long when TRIX crosses above zero AND volume confirms AND CHOP > 61.8 (mean reversion in range).
Short when TRIX crosses below zero AND volume confirms AND CHOP > 61.8.
Exit when TRIX crosses zero in opposite direction OR CHOP < 38.2 (trending regime).
Designed for 4h timeframe to achieve 75-200 total trades over 4 years with low fee drag.
Works in both bull and bear markets by adapting to regime: mean revert in chop, trend follow in strong trends.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for choppiness regime filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate TRIX: 15-period EMA of EMA of EMA of close
    close_s = pd.Series(close)
    ema1 = close_s.ewm(span=15, adjust=False, min_periods=15).mean()
    ema2 = ema1.ewm(span=15, adjust=False, min_periods=15).mean()
    ema3 = ema2.ewm(span=15, adjust=False, min_periods=15).mean()
    trix = 100 * (ema3.pct_change())
    trix_values = trix.values
    trix_prev = np.roll(trix_values, 1)
    trix_prev[0] = np.nan
    
    # Align TRIX to 4h timeframe (no additional delay needed for EMA-based)
    trix_aligned = align_htf_to_ltf(prices, df_1d, trix_values)
    trix_prev_aligned = align_htf_to_ltf(prices, df_1d, trix_prev)
    
    # Calculate 1d Choppiness Index: CHOP = 100 * log10(sum(ATR(14)) / (log10(n) * (max(high)-min(low))))
    # Simplified: CHOP = 100 * log10(ATR_sum / (log10(14) * (HHV - LLV))) where ATR_sum is sum of TR over 14 periods
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # first TR is undefined
    
    # ATR(14) = smoothed TR
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Sum of ATR over 14 periods
    atr_sum = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index
    chop = 100 * np.log10(atr_sum / (np.log10(14) * (hh - ll)))
    chop_values = chop
    
    # Align Chop to 4h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_values)
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need TRIX (45), Chop (14+14), volume avg (20)
    start_idx = max(45, 28, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(trix_aligned[i]) or np.isnan(trix_prev_aligned[i]) or 
            np.isnan(chop_aligned[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        trix_val = trix_aligned[i]
        trix_prev_val = trix_prev_aligned[i]
        chop_val = chop_aligned[i]
        vol_conf = volume_confirm[i]
        
        if position == 0:
            # Look for entry: TRIX zero-cross with volume confirmation AND chop regime (rangy > 61.8)
            # Long: TRIX crosses above zero AND volume confirms AND chop > 61.8 (mean reversion in range)
            long_condition = (trix_prev_val <= 0) and (trix_val > 0) and vol_conf and (chop_val > 61.8)
            # Short: TRIX crosses below zero AND volume confirms AND chop > 61.8
            short_condition = (trix_prev_val >= 0) and (trix_val < 0) and vol_conf and (chop_val > 61.8)
            
            if long_condition:
                signals[i] = size
                position = 1
            elif short_condition:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long when TRIX crosses below zero OR chop regime shifts to trending (< 38.2)
            exit_condition = (trix_val < 0) or (chop_val < 38.2)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short when TRIX crosses above zero OR chop regime shifts to trending (< 38.2)
            exit_condition = (trix_val > 0) or (chop_val < 38.2)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_TRIX_VolumeSpike_ChopRegime_Trend"
timeframe = "4h"
leverage = 1.0