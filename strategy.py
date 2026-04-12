#!/usr/bin/env python3
"""
4h_12h_trix_volume_reversal_v1
Hypothesis: 4-hour TRIX momentum with volume spike and 12h trend filter.
Enters long when TRIX crosses above zero with volume spike and 12h uptrend; short when crosses below zero with volume spike and 12h downtrend.
Uses fixed position sizing to minimize churn. Designed to catch momentum reversals in trending markets.
Target: 20-35 trades/year (80-140 total over 4 years) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate TRIX (15-period EMA of EMA of EMA of close, then ROC)
    ema1 = pd.Series(close).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean().values
    trix = 100 * (pd.Series(ema3).pct_change(1).values)
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # 12h EMA50 for trend direction
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Calculate ATR for volatility filter
    high = prices['high'].values
    low = prices['low'].values
    tr1 = np.abs(high - low)
    tr2 = np.abs(np.roll(high, 1) - close)
    tr3 = np.abs(np.roll(low, 1) - close)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(trix[i]) or np.isnan(ema50_12h_aligned[i]) or 
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period average
        if i >= 20:
            vol_ma = np.mean(volume[max(0, i-20):i])
            volume_filter = volume[i] > vol_ma * 1.5
        else:
            volume_filter = False
        
        # Trend filter from 12h EMA50
        uptrend_12h = close[i] > ema50_12h_aligned[i]
        downtrend_12h = close[i] < ema50_12h_aligned[i]
        
        # Fixed position size
        position_size = 0.25
        
        # TRIX zero cross signals
        trix_cross_up = trix[i] > 0 and trix[i-1] <= 0
        trix_cross_down = trix[i] < 0 and trix[i-1] >= 0
        
        # Entry conditions: TRIX cross with volume and trend confirmation
        long_entry = trix_cross_up and volume_filter and uptrend_12h
        short_entry = trix_cross_down and volume_filter and downtrend_12h
        
        # Exit conditions: opposite TRIX cross or trend change
        long_exit = trix_cross_down or not uptrend_12h
        short_exit = trix_cross_up or not downtrend_12h
        
        if long_entry and position != 1:
            position = 1
            signals[i] = position_size
        elif short_entry and position != -1:
            position = -1
            signals[i] = -position_size
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_12h_trix_volume_reversal_v1"
timeframe = "4h"
leverage = 1.0