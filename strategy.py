#!/usr/bin/env python3
"""
6h_Trix_15_Signal_MeanReversion
Hypothesis: TRIX (triple-smoothed EMA) identifies overbought/oversold conditions.
In ranging markets (6h), TRIX > 0.15 signals overextended long, TRIX < -0.15 signals overextended short.
Mean reversion to TRIX=0 works in both bull and bear regimes as price oscillates around mean.
Uses 1d trend filter: only take long signals when price > 1d EMA50, short when price < 1d EMA50.
Targets 15-30 trades/year to minimize fee decay while capturing mean reversion swings.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_trix(close, period=15):
    """Calculate TRIX: triple EMA of log returns."""
    # Calculate log returns
    log_close = np.log(close)
    # First EMA
    ema1 = pd.Series(log_close).ewm(span=period, adjust=False, min_periods=period).mean().values
    # Second EMA
    ema2 = pd.Series(ema1).ewm(span=period, adjust=False, min_periods=period).mean().values
    # Third EMA
    ema3 = pd.Series(ema2).ewm(span=period, adjust=False, min_periods=period).mean().values
    # TRIX = percentage change of third EMA
    trix = np.diff(ema3, prepend=ema3[0]) / ema3[:-1] * 100
    # Handle first element
    trix = np.insert(trix, 0, 0.0)
    return trix

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate TRIX on 6h close
    trix = calculate_trix(close, 15)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 50  # Warmup for TRIX and EMA
    
    for i in range(start_idx, n):
        if np.isnan(trix[i]) or np.isnan(ema_50_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        trix_val = trix[i]
        ema50 = ema_50_1d_aligned[i]
        
        if position == 0:
            # Long: TRIX oversold (< -0.15) and price above 1d EMA50 (uptrend filter)
            if trix_val < -0.15 and price > ema50:
                signals[i] = 0.25
                position = 1
            # Short: TRIX overbought (> 0.15) and price below 1d EMA50 (downtrend filter)
            elif trix_val > 0.15 and price < ema50:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: TRIX returns to zero (mean reversion) or trend fails
            if trix_val >= 0.0 or price < ema50:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: TRIX returns to zero (mean reversion) or trend fails
            if trix_val <= 0.0 or price > ema50:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Trix_15_Signal_MeanReversion"
timeframe = "6h"
leverage = 1.0