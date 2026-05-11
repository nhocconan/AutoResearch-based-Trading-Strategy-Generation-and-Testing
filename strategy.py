#!/usr/bin/env python3
# 4h_1d_TRIX_ZeroCross_Volume_Confirm
# Hypothesis: TRIX (triple-smoothed EMA) zero cross signals momentum shifts.
# Combined with 1d trend filter (EMA50) and volume confirmation (2x 20-period avg).
# Trades only when TRIX crosses zero AND price is on correct side of 1d EMA50.
# Volume surge filters breakouts with conviction. Target: ~25 trades/year.

name = "4h_1d_TRIX_ZeroCross_Volume_Confirm"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 1-day data for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 4h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- 1d EMA50 for trend filter ---
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # --- TRIX on 4h (triple EMA 12-period) ---
    # EMA1
    ema1 = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).mean().values
    # EMA2 of EMA1
    ema2 = pd.Series(ema1).ewm(span=12, adjust=False, min_periods=12).mean().values
    # EMA3 of EMA2 (TRIX)
    ema3 = pd.Series(ema2).ewm(span=12, adjust=False, min_periods=12).mean().values
    # TRIX = (EMA3[t] - EMA3[t-1]) / EMA3[t-1] * 100
    trix = np.zeros_like(close)
    trix[1:] = (ema3[1:] - ema3[:-1]) / ema3[:-1] * 100
    
    # --- Volume confirmation (2x 20-period average) ---
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: enough for TRIX calculation (36 periods for 3x EMA12) and volume MA
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(trix[i]) or
            np.isnan(trix[i-1]) or
            np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        volume_surge = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long: TRIX crosses above zero with volume surge and 1d uptrend
            if (trix[i] > 0 and trix[i-1] <= 0 and 
                volume_surge and 
                ema_50_1d_aligned[i] < close[i]):
                signals[i] = 0.25
                position = 1
            # Short: TRIX crosses below zero with volume surge and 1d downtrend
            elif (trix[i] < 0 and trix[i-1] >= 0 and 
                  volume_surge and 
                  ema_50_1d_aligned[i] > close[i]):
                signals[i] = -0.25
                position = -1
        else:
            if position == 1:
                # Exit long: TRIX crosses below zero OR price below 1d EMA50
                if (trix[i] < 0 and trix[i-1] >= 0) or (close[i] < ema_50_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: TRIX crosses above zero OR price above 1d EMA50
                if (trix[i] > 0 and trix[i-1] <= 0) or (close[i] > ema_50_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals