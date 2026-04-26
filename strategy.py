#!/usr/bin/env python3
"""
4h_TRIX_VolumeSpike_ChopRegime_TrendFilter_v1
Hypothesis: TRIX (15-period) crossing zero line with volume spike confirmation, choppiness regime filter (CHOP > 61.8 = range), and 4h EMA50 trend filter. Designed to capture momentum reversals in ranging markets while avoiding whipsaws in strong trends. Uses discrete position sizing (0.25) to minimize fee churn. Works in both bull and bear markets by adapting to regime.
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
    
    # Get 4h data for EMA50 trend filter and choppiness regime
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Choppiness Index on 4h (14-period)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h_arr = df_4h['close'].values
    
    # True Range
    tr1 = high_4h[1:] - low_4h[1:]
    tr2 = np.abs(high_4h[1:] - close_4h_arr[:-1])
    tr3 = np.abs(low_4h[1:] - close_4h_arr[:-1])
    tr_4h = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_4h = pd.Series(tr_4h).rolling(window=14, min_periods=14).mean().values
    
    # Sum of ATR over 14 periods
    sum_atr_14 = pd.Series(atr_4h).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    highest_high_14 = pd.Series(high_4h).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_4h).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index: CHOP = 100 * log10(sum(ATR14) / (HH14 - LL14)) / log10(14)
    range_14 = highest_high_14 - lowest_low_14
    # Avoid division by zero
    range_14 = np.where(range_14 == 0, 1e-10, range_14)
    chop_4h = 100 * np.log10(sum_atr_14 / range_14) / np.log10(14)
    chop_4h_aligned = align_htf_to_ltf(prices, df_4h, chop_4h)
    
    # TRIX (15-period) on 15m close prices
    # TRIX = EMA(EMA(EMA(close, 15), 15), 15) - then percent change
    ema1 = pd.Series(close).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean().values
    trix_raw = pd.Series(ema3).pct_change(periods=1) * 100  # Percentage change
    trix = trix_raw.values
    
    # Volume confirmation: 2.0x average volume (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: max of TRIX (15*3=45 for stability), EMA50 (50), CHOP (14*2=28), volume MA (20)
    start_idx = max(45, 50, 28, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(trix[i]) or 
            np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(chop_4h_aligned[i]) or 
            np.isnan(vol_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        trix_val = trix[i]
        ema_50_4h_val = ema_50_4h_aligned[i]
        chop_val = chop_4h_aligned[i]
        close_val = close[i]
        volume_val = volume[i]
        vol_ma_val = vol_ma[i]
        
        if position == 0:
            # Long: TRIX crosses above zero with volume spike and in ranging market (CHOP > 61.8)
            # Avoid long in strong uptrend (price > EMA50) to prevent buying tops
            trix_cross_up = trix_val > 0 and trix[i-1] <= 0
            long_signal = (trix_cross_up and 
                          volume_val > 2.0 * vol_ma_val and 
                          chop_val > 61.8 and 
                          close_val < ema_50_4h_val)  # Not in strong uptrend
            
            # Short: TRIX crosses below zero with volume spike and in ranging market (CHOP > 61.8)
            # Avoid short in strong downtrend (price < EMA50) to prevent selling bottoms
            trix_cross_down = trix_val < 0 and trix[i-1] >= 0
            short_signal = (trix_cross_down and 
                           volume_val > 2.0 * vol_ma_val and 
                           chop_val > 61.8 and 
                           close_val > ema_50_4h_val)  # Not in strong downtrend
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: TRIX crosses below zero OR price breaks above EMA50 (trend change)
            if (trix_val < 0 and trix[i-1] >= 0) or close_val > ema_50_4h_val:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: TRIX crosses above zero OR price breaks below EMA50 (trend change)
            if (trix_val > 0 and trix[i-1] <= 0) or close_val < ema_50_4h_val:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_TRIX_VolumeSpike_ChopRegime_TrendFilter_v1"
timeframe = "4h"
leverage = 1.0