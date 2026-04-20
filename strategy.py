#!/usr/bin/env python3
# 1d_1w_MomentumBreakout_WithTrendFilter_V1
# Hypothesis: Daily momentum breakouts (price > 5-day high) with weekly trend filter (price > 20-week EMA)
# and volume confirmation (2x average) capture sustained moves in both bull and bear markets.
# Exit when price closes below 5-day low (long) or above 5-day high (short).
# Target: 15-25 trades per year per symbol to avoid fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_MomentumBreakout_WithTrendFilter_V1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # === Weekly EMA(20) for trend filter ===
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # === Daily indicators ===
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily 5-period high/low for breakout
    high_5 = pd.Series(high).rolling(window=5, min_periods=5).max().values
    low_5 = pd.Series(low).rolling(window=5, min_periods=5).min().values
    
    # Volume ratio (current vs 20-day average)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after warmup
        # Get values
        close_val = close[i]
        high_5_val = high_5[i]
        low_5_val = low_5[i]
        ema_20_1w_val = ema_20_1w_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if (np.isnan(high_5_val) or np.isnan(low_5_val) or 
            np.isnan(ema_20_1w_val) or np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above 5-day high, above weekly EMA20, with volume confirmation
            if close_val > high_5_val and close_val > ema_20_1w_val and vol_ratio_val > 2.0:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below 5-day low, below weekly EMA20, with volume confirmation
            elif close_val < low_5_val and close_val < ema_20_1w_val and vol_ratio_val > 2.0:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price closes below 5-day low
            if close_val < low_5_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price closes above 5-day high
            if close_val > high_5_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals