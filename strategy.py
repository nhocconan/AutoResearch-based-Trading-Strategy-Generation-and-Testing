#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h TRIX(9) zero-line crossover with 1d EMA50 trend filter and volume confirmation.
# TRIX filters noise and identifies momentum shifts. Zero-line cross signals trend changes.
# Long when TRIX crosses above zero in bull trend (close > 1d EMA50) with volume > 1.8x 20-period MA.
# Short when TRIX crosses below zero in bear trend (close < 1d EMA50) with volume spike.
# Exit on opposite TRIX zero-line cross or trend reversal.
# Target: 50-150 total trades over 4 years (12-37/year) with discrete sizing 0.25.
# Works in both bull and bear by following 1d trend direction and using momentum confirmation.

name = "12h_TRIX_ZeroCross_1dEMA50_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate TRIX(9) on 12h close: triple EMA then ROC
    ema1 = pd.Series(close).ewm(span=9, min_periods=9, adjust=False).mean()
    ema2 = ema1.ewm(span=9, min_periods=9, adjust=False).mean()
    ema3 = ema2.ewm(span=9, min_periods=9, adjust=False).mean()
    trix = (ema3.pct_change() * 100).values  # Percentage change
    
    # Volume regime: current 12h volume > 1.8x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(trix[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        ema_trend = ema_50_1d_aligned[i]
        trix_val = trix[i]
        trix_prev = trix[i-1] if i > 0 else 0
        vol_spike = volume_spike[i]
        
        # Determine trend regime
        is_bull_trend = close_val > ema_trend
        is_bear_trend = close_val < ema_trend
        
        # TRIX zero-line cross detection
        trix_cross_above = trix_prev <= 0 and trix_val > 0
        trix_cross_below = trix_prev >= 0 and trix_val < 0
        
        # Entry logic
        if position == 0:
            if is_bull_trend and trix_cross_above and vol_spike:
                signals[i] = 0.25
                position = 1
            elif is_bear_trend and trix_cross_below and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: TRIX crosses below zero OR trend reversal
            if trix_cross_below or close_val < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: TRIX crosses above zero OR trend reversal
            if trix_cross_above or close_val > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals