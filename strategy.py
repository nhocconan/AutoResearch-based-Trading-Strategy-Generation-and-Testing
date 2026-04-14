#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Turtle Trading Breakout with Daily Volatility Filter
# Uses Donchian breakout (20-period) for entries with daily ATR filter to avoid false breakouts
# Position sizing based on volatility (risk parity) to maintain consistent risk
# Works in both bull and bear markets by capturing breakouts in direction of trend
# Target: 20-30 trades/year (80-120 total over 4 years) to minimize fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Load daily data ONCE before loop for ATR filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily ATR (14) for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = np.nan
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate Donchian channels (20-period) on 6h data
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0
    
    # Start after enough data for calculations
    start = 20  # for Donchian channels
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(atr_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long breakout: price breaks above 20-period high with volatility filter
            if high[i] > highest_high[i] and atr_1d_aligned[i] > 0:
                # Size inversely proportional to volatility (risk parity)
                vol_scaled = min(0.30, 0.02 / atr_1d_aligned[i] * 100)  # Scale to reasonable size
                position = 1
                signals[i] = vol_scaled
            # Short breakdown: price breaks below 20-period low with volatility filter
            elif low[i] < lowest_low[i] and atr_1d_aligned[i] > 0:
                vol_scaled = min(0.30, 0.02 / atr_1d_aligned[i] * 100)
                position = -1
                signals[i] = -vol_scaled
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below 10-period low (stop and reverse)
            low_10 = pd.Series(low).rolling(window=10, min_periods=10).min().values[i]
            if not np.isnan(low_10) and low[i] < low_10:
                position = -1
                vol_scaled = min(0.30, 0.02 / atr_1d_aligned[i] * 100)
                signals[i] = -vol_scaled  # Reverse and go short
            else:
                signals[i] = position  # Maintain position
        elif position == -1:
            # Exit short: price breaks above 10-period high (stop and reverse)
            high_10 = pd.Series(high).rolling(window=10, min_periods=10).max().values[i]
            if not np.isnan(high_10) and high[i] > high_10:
                position = 1
                vol_scaled = min(0.30, 0.02 / atr_1d_aligned[i] * 100)
                signals[i] = vol_scaled  # Reverse and go long
            else:
                signals[i] = position  # Maintain position
    
    return signals

name = "6h_Turtle_Breakout_DailyATR_Filter"
timeframe = "6h"
leverage = 1.0