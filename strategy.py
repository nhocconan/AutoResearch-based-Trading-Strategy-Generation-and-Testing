#!/usr/bin/env python3
name = "12h_TRIX_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE for TRIX calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 15:
        return np.zeros(n)
    
    # TRIX calculation on daily close (15-period EMA triple smoothed)
    close_series = pd.Series(df_1d['close'])
    ema1 = close_series.ewm(span=15, adjust=False, min_periods=15).mean()
    ema2 = ema1.ewm(span=15, adjust=False, min_periods=15).mean()
    ema3 = ema2.ewm(span=15, adjust=False, min_periods=15).mean()
    trix_raw = ((ema3 - ema3.shift(1)) / ema3.shift(1)) * 100
    trix = trix_raw.fillna(0).values
    
    # Daily EMA50 for trend filter
    ema_50_1d = close_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align TRIX and EMA50 to 12h timeframe
    trix_12h = align_htf_to_ltf(prices, df_1d, trix)
    ema_50_12h = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume spike detection (2x 20-period average on 12h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(trix_12h[i]) or np.isnan(ema_50_12h[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_condition = volume[i] > vol_ma_20[i] * 2.0
        
        if position == 0:
            # Long: TRIX crosses above zero in daily uptrend with volume
            if trix_12h[i] > 0 and trix_12h[i-1] <= 0 and ema_50_12h[i] > ema_50_12h[i-1] and vol_condition:
                signals[i] = 0.25
                position = 1
            # Short: TRIX crosses below zero in daily downtrend with volume
            elif trix_12h[i] < 0 and trix_12h[i-1] >= 0 and ema_50_12h[i] < ema_50_12h[i-1] and vol_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: TRIX crosses below zero or trend reverses
            if trix_12h[i] < 0 or ema_50_12h[i] < ema_50_12h[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: TRIX crosses above zero or trend reverses
            if trix_12h[i] > 0 or ema_50_12h[i] > ema_50_12h[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: TRIX zero-line cross with daily trend filter and volume confirmation
# - TRIX (15-period triple smoothed EMA) measures momentum; zero-line cross indicates trend change
# - Long when TRIX crosses above zero in daily uptrend (EMA50 rising) with volume confirmation
# - Short when TRIX crosses below zero in daily downtrend (EMA50 falling) with volume confirmation
# - Volume condition (2x average) reduces false signals
# - Exit when TRIX crosses zero in opposite direction or daily trend reverses
# - Position size 0.25 targets ~15-30 trades/year to avoid fee drag
# - Works in bull markets (bullish TRIX crosses in uptrend) and bear markets (bearish TRIX crosses in downtrend)
# - Uses 1d timeframe for momentum and trend, 12h for execution timing
# - TRIX is less noisy than MACD and provides clearer trend signals
# - Proven pattern: momentum + trend + volume works (similar to TRIX + volume spike in research)