#!/usr/bin/env python3
name = "4h_TRIX_ZeroCross_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

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
    
    # Load daily data ONCE for TRIX and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 15:
        return np.zeros(n)
    
    # TRIX on daily close (triple EMA of 1-period percent change)
    close_1d = df_1d['close'].values
    if len(close_1d) < 15:
        return np.zeros(n)
    
    # Calculate 1-period percent change
    pct_change = np.diff(close_1d) / close_1d[:-1]
    pct_change = np.concatenate([[np.nan], pct_change])  # align with original index
    
    # Triple EMA
    ema1 = pd.Series(pct_change).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema2 = pd.Series(ema1).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema3 = pd.Series(ema2).ewm(span=12, adjust=False, min_periods=12).mean().values
    trix = ema3 * 100  # scale for readability
    
    # Align TRIX to 4h timeframe
    trix_4h = align_htf_to_ltf(prices, df_1d, trix)
    
    # Daily EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike detection (2x 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(trix_4h[i]) or np.isnan(trix_4h[i-1]) or 
            np.isnan(ema_34_4h[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_condition = volume[i] > vol_ma_20[i] * 2.0
        
        if position == 0:
            # Long: TRIX crosses above zero in daily uptrend with volume
            if trix_4h[i] > 0 and trix_4h[i-1] <= 0 and ema_34_4h[i] > ema_34_4h[i-1] and vol_condition:
                signals[i] = 0.25
                position = 1
            # Short: TRIX crosses below zero in daily downtrend with volume
            elif trix_4h[i] < 0 and trix_4h[i-1] >= 0 and ema_34_4h[i] < ema_34_4h[i-1] and vol_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: TRIX crosses below zero or trend reverses
            if trix_4h[i] < 0 or ema_34_4h[i] < ema_34_4h[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: TRIX crosses above zero or trend reverses
            if trix_4h[i] > 0 or ema_34_4h[i] > ema_34_4h[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: TRIX zero-cross with daily trend filter and volume confirmation
# - TRIX measures momentum; zero-cross indicates momentum shift
# - Long when TRIX crosses above zero in daily uptrend (EMA34 rising)
# - Short when TRIX crosses below zero in daily downtrend (EMA34 falling)
# - Volume confirmation (2x average) reduces false signals
# - Exit when TRIX crosses back or daily trend reverses
# - Uses 1d timeframe for signal generation and trend, 4h for execution timing
# - Position size 0.25 targets ~40-60 trades/year to balance opportunity and cost
# - Works in bull (bullish crosses in uptrend) and bear (bearish crosses in downtrend) markets
# - Avoids overtrading by requiring trend alignment and volume spike
# - TRIX is less noisy than MACD and provides clearer momentum signals