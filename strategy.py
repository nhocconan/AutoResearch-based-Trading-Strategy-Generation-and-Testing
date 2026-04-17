#!/usr/bin/env python3
"""
4h_TRIX_VolumeSpike_TrendFilter
Strategy: TRIX momentum with volume spike confirmation and 1d EMA trend filter.
Long: TRIX > 0 + volume > 2.0x average + price > 1d EMA50
Short: TRIX < 0 + volume > 2.0x average + price < 1d EMA50
Exit: TRIX crosses zero or trend fails
Position size: 0.25
Designed to capture momentum bursts with institutional volume in trending markets.
Timeframe: 4h
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate TRIX (15-period EMA of EMA of EMA)
    close_series = pd.Series(close)
    ema1 = close_series.ewm(span=15, adjust=False, min_periods=15).mean()
    ema2 = ema1.ewm(span=15, adjust=False, min_periods=15).mean()
    ema3 = ema2.ewm(span=15, adjust=False, min_periods=15).mean()
    trix = ema3.pct_change() * 100  # Percentage change
    trix_values = tria.values if hasattr(trix, 'values') else trix.values
    
    # Volume confirmation (20-period average)
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Get 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    close_series_1d = pd.Series(close_1d)
    ema50_1d = close_series_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Need enough data for EMA50 and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(trix_values[i]) or np.isnan(volume_ma20[i]) or 
            np.isnan(ema50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 2.0x 20-period average
        volume_filter = volume[i] > (2.0 * volume_ma20[i])
        
        # Trend filter: price > 1d EMA50 for long, < for short
        price_above_ema50 = close[i] > ema50_1d_aligned[i]
        price_below_ema50 = close[i] < ema50_1d_aligned[i]
        
        # TRIX signals
        trix_positive = trix_values[i] > 0
        trix_negative = trix_values[i] < 0
        
        # Entry conditions
        if position == 0:
            # Long: TRIX positive + volume spike + price above EMA50
            if trix_positive and volume_filter and price_above_ema50:
                signals[i] = 0.25
                position = 1
            # Short: TRIX negative + volume spike + price below EMA50
            elif trix_negative and volume_filter and price_below_ema50:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: TRIX turns negative or trend fails
            if not trix_positive or not price_above_ema50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: TRIX turns positive or trend fails
            if not trix_negative or not price_below_ema50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_TRIX_VolumeSpike_TrendFilter"
timeframe = "4h"
leverage = 1.0