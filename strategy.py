#!/usr/bin/env python3
"""
4h_TRIX_ZeroCross_12hTrend_VolumeSpike
Hypothesis: On 4h timeframe, enter long when TRIX(12) crosses above zero and 12h EMA50 trend is bullish, confirmed by volume spike (>1.5x 20-bar MA). Enter short when TRIX crosses below zero and 12h EMA50 is bearish with volume confirmation. Uses TRIX for momentum reversal detection, 12h HTF for trend alignment, and volume to filter false signals. Designed for 20-50 trades/year (80-200 total over 4 years) to avoid fee drag. Works in both bull and bear markets by following the 12h trend while using TRIX zero-cross for precise entries.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 60:
        return np.zeros(n)
    
    # 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # TRIX calculation: triple EMA of ROC
    # ROC = (close / close.shift(1) - 1) * 100
    close_series = pd.Series(close)
    roc = (close_series / close_series.shift(1) - 1) * 100
    roc_values = roc.values
    
    # Triple EMA: EMA(EMA(EMA(ROC)))
    ema1 = pd.Series(roc_values).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema2 = pd.Series(ema1).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema3 = pd.Series(ema2).ewm(span=12, adjust=False, min_periods=12).mean().values
    trix = ema3  # TRIX is the final smoothed value
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25  # Position size
    
    # Warmup: max of calculations (12*3 for TRIX, 20 for vol, 50 for ema)
    start_idx = max(36, 20, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(trix[i]) or 
            np.isnan(trix[i-1]) or 
            np.isnan(vol_ma[i])):
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        close_val = close[i]
        ema_50_val = ema_50_12h_aligned[i]
        trix_now = trix[i]
        trix_prev = trix[i-1]
        vol_spike = volume_spike[i]
        
        # Determine 12h trend: bullish if price > EMA50, bearish if price < EMA50
        bullish_12h = close_val > ema_50_val
        bearish_12h = close_val < ema_50_val
        
        # TRIX zero-cross signals
        trix_cross_up = (trix_prev <= 0) and (trix_now > 0)   # Bullish momentum
        trix_cross_down = (trix_prev >= 0) and (trix_now < 0) # Bearish momentum
        
        # Entry conditions: TRIX zero-cross in trend direction with volume
        long_entry = trix_cross_up and bullish_12h and vol_spike
        short_entry = trix_cross_down and bearish_12h and vol_spike
        
        # Exit conditions: opposite TRIX cross or trend reversal
        exit_long = trix_cross_down or not bullish_12h
        exit_short = trix_cross_up or not bearish_12h
        
        if long_entry and position != 1:
            signals[i] = base_size
            position = 1
        elif short_entry and position != -1:
            signals[i] = -base_size
            position = -1
        elif position == 1 and exit_long:
            signals[i] = 0.0
            position = 0
        elif position == -1 and exit_short:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
    
    return signals

name = "4h_TRIX_ZeroCross_12hTrend_VolumeSpike"
timeframe = "4h"
leverage = 1.0