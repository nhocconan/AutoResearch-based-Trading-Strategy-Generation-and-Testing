#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_TRIX_10_Trend_Filter_1dV"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data once for TRIX and volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # TRIX: Triple Exponential Moving Average
    close_1d = df_1d['close'].values
    
    # Triple EMA calculation
    ema1 = pd.Series(close_1d).ewm(span=10, adjust=False, min_periods=10).mean()
    ema2 = ema1.ewm(span=10, adjust=False, min_periods=10).mean()
    ema3 = ema2.ewm(span=10, adjust=False, min_periods=10).mean()
    trix_raw = ((ema3 / ema3.shift(1)) - 1) * 100
    trix = trix_raw.fillna(0).values
    
    # TRIX signal line (9-period EMA of TRIX)
    trix_signal = pd.Series(trix).ewm(span=9, adjust=False, min_periods=9).mean().values
    
    # Daily volume filter: volume > 1.5 * 20-period average
    vol_ma = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_filter = df_1d['volume'].values > (vol_ma * 1.5)
    
    # Align TRIX, signal, and volume filter to 4h timeframe
    trix_aligned = align_htf_to_ltf(prices, df_1d, trix)
    trix_signal_aligned = align_htf_to_ltf(prices, df_1d, trix_signal)
    vol_filter_aligned = align_htf_to_ltf(prices, df_1d, vol_filter.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if np.isnan(trix_aligned[i]) or np.isnan(trix_signal_aligned[i]) or np.isnan(vol_filter_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: TRIX crosses above signal line AND volume filter active
            long_cond = (trix_aligned[i] > trix_signal_aligned[i]) and (trix_aligned[i-1] <= trix_signal_aligned[i-1]) and (vol_filter_aligned[i] > 0.5)
            
            # Short entry: TRIX crosses below signal line AND volume filter active
            short_cond = (trix_aligned[i] < trix_signal_aligned[i]) and (trix_aligned[i-1] >= trix_signal_aligned[i-1]) and (vol_filter_aligned[i] > 0.5)
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: TRIX crosses below signal line
            if trix_aligned[i] < trix_signal_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: TRIX crosses above signal line
            if trix_aligned[i] > trix_signal_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: TRIX (Triple Exponential Moving Average) identifies momentum shifts with reduced noise.
# Long when TRIX crosses above its signal line with volume confirmation (volume > 1.5x 20-day average).
# Short when TRIX crosses below its signal line with volume confirmation.
# Uses daily TRIX for stable signal generation, aligned to 4h timeframe.
# Volume filter ensures trades occur during periods of increased market participation.
# Designed for 4h timeframe to target 20-50 trades per year, minimizing fee drag.
# Works in both bull and bear markets by capturing momentum reversals.