#!/usr/bin/env python3
# 4h_TRIX_ZeroCross_Volume_Spike
# Hypothesis: TRIX (1-period rate of change of triple-smoothed EMA) crossing zero indicates momentum shift.
# Combined with volume spike (>2x 20-bar average) for confirmation, this captures strong momentum moves.
# In bull markets: zero-cross up + volume surge = long. In bear markets: zero-cross down + volume surge = short.
# Uses 1d EMA200 as trend filter to avoid counter-trend trades. Designed for low trade frequency (~20-40/year)
# to minimize fee drag while capturing significant momentum shifts.

name = "4h_TRIX_ZeroCross_Volume_Spike"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 4h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- 1d EMA200 for trend filter ---
    ema_200_1d = pd.Series(df_1d['close'].values).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # --- TRIX on 4h (15-period) ---
    # TRIX = 100 * (EMA3 of EMA2 of EMA1 of close - previous EMA3) / previous EMA3
    # EMA1
    ema1 = pd.Series(close).ewm(span=15, adjust=False, min_periods=15).mean().values
    # EMA2
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean().values
    # EMA3
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean().values
    # TRIX
    trix = 100 * (ema3 - np.roll(ema3, 1)) / np.roll(ema3, 1)
    trix[0] = 0  # first value undefined
    
    # --- Volume confirmation (2.0x 20-period average on 4h) ---
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: enough for TRIX (3*15=45) and volume MA (20)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(trix[i]) or
            np.isnan(ema_200_1d_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        volume_surge = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long: TRIX crosses above zero with volume surge and 1d EMA200 uptrend
            if trix[i] > 0 and trix[i-1] <= 0 and volume_surge and ema_200_1d_aligned[i] < close[i]:
                signals[i] = 0.25
                position = 1
            # Short: TRIX crosses below zero with volume surge and 1d EMA200 downtrend
            elif trix[i] < 0 and trix[i-1] >= 0 and volume_surge and ema_200_1d_aligned[i] > close[i]:
                signals[i] = -0.25
                position = -1
        else:
            if position == 1:
                # Exit long: TRIX crosses below zero OR price crosses below 1d EMA200
                if trix[i] < 0 and trix[i-1] >= 0 or close[i] < ema_200_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: TRIX crosses above zero OR price crosses above 1d EMA200
                if trix[i] > 0 and trix[i-1] <= 0 or close[i] > ema_200_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals