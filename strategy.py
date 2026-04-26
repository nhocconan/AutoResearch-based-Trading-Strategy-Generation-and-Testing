#!/usr/bin/env python3
"""
6h_WilliamsVixFix_ExtremeReversion_1dTrendFilter_v1
Hypothesis: Williams Vix Fix (WVF) identifies extreme fear/greed on 6h. When WVF > 0.8 (extreme fear) during 1d uptrend → long reversal; when WVF < 0.2 (extreme greed) during 1d downtrend → short reversal. Uses 1d EMA50 as trend filter to ensure reversals align with higher timeframe momentum. Targets 15-25 trades/year with discrete sizing (0.25) to minimize fee drag. Designed for mean reversion in extended moves, works in both bull/bear by only trading counter-trend extreme reversals with trend alignment.
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
    
    # Get 1d data for HTF trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Williams Vix Fix: measures market fear (0-1, higher = more fear)
    # WVF = ((Highest Close in Period - Low) / (Highest Close in Period - Lowest Close in Period)) * 100
    lookback = 22  # Standard WVF lookback
    highest_close = pd.Series(close).rolling(window=lookback, min_periods=lookback).max().values
    lowest_close = pd.Series(close).rolling(window=lookback, min_periods=lookback).min().values
    
    # Avoid division by zero
    denominator = highest_close - lowest_close
    denominator = np.where(denominator == 0, 1e-10, denominator)
    
    wvf = ((highest_close - low) / denominator) * 100
    # Normalize to 0-1 scale (typical WVF ranges 0-100+)
    wvf_normalized = np.clip(wfv / 100.0, 0, 1) if 'wfv' in locals() else np.clip(wvf / 100.0, 0, 1)
    wvf_normalized = np.clip(wvf / 100.0, 0, 1)
    
    # Align HTF indicators to 6h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    wvf_aligned = align_htf_to_ltf(prices, df_1d, wvf_normalized)  # WVF uses same lookback, so 1d alignment
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max of EMA(50) 1d, WVF lookback (22)
    start_idx = max(50, lookback) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(wvf_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        ema_50_1d_val = ema_50_1d_aligned[i]
        close_val = close[i]
        wvf_val = wvf_aligned[i]
        
        # Trend filter: 1d EMA50 direction
        uptrend = close_val > ema_50_1d_val
        downtrend = close_val < ema_50_1d_val
        
        # Extreme fear/greed thresholds
        extreme_fear = wvf_val > 0.80   # WVF > 80 = extreme fear
        extreme_greed = wvf_val < 0.20  # WVF < 20 = extreme greed
        
        if position == 0:
            # Long reversal: extreme fear during 1d uptrend (buy panic dips in uptrend)
            long_signal = extreme_fear and uptrend
            
            # Short reversal: extreme greed during 1d downtrend (sell euphoric rallies in downtrend)
            short_signal = extreme_greed and downtrend
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long: exit when fear subsides or trend breaks
            signals[i] = 0.25
            # Exit conditions: fear reduced OR price breaks below 1d EMA50
            if wvf_val < 0.50 or close_val < ema_50_1d_val:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short: exit when greed subsides or trend breaks
            signals[i] = -0.25
            # Exit conditions: greed reduced OR price breaks above 1d EMA50
            if wvf_val > 0.50 or close_val > ema_50_1d_val:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_WilliamsVixFix_ExtremeReversion_1dTrendFilter_v1"
timeframe = "6h"
leverage = 1.0