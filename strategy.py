#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily 200-period EMA defines long-term trend. Price pullbacks to EMA in trending markets offer high-probability entries.
# Weekly trend filter (price vs weekly EMA50) ensures alignment with higher timeframe momentum.
# Volume confirmation (1.5x 20-day average) filters low-conviction moves.
# Designed for fewer trades (<50/year) to minimize fee drag and work in both bull/bear markets via trend alignment.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily EMA200 for trend and pullback signals
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    close_1d_series = pd.Series(close_1d)
    ema200_1d = close_1d_series.ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Weekly EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    close_1w_series = pd.Series(close_1w)
    ema50_1w = close_1w_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume filter: current volume > 1.5x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 200  # Ensure EMA200 is valid
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema200_1d_aligned[i]) or np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above weekly EMA50 (uptrend) and pulls back to touch or cross above daily EMA200 with volume
            if close[i] > ema50_1w_aligned[i] and close[i] >= ema200_1d_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price below weekly EMA50 (downtrend) and pulls back to touch or cross below daily EMA200 with volume
            elif close[i] < ema50_1w_aligned[i] and close[i] <= ema200_1d_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses back below daily EMA200
            if close[i] < ema200_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses back above daily EMA200
            if close[i] > ema200_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_EMA200_Pullback_WeeklyTrend_Filter"
timeframe = "1d"
leverage = 1.0