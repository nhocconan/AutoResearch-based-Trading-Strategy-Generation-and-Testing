#!/usr/bin/env python3
name = "6H_Daily_Trix_Trend_Reversal"
timeframe = "6h"
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
    
    # Get daily data for TRIX and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    # Calculate daily TRIX (15-period EMA of EMA of EMA of close, then ROC)
    close_series = pd.Series(df_1d['close'])
    ema1 = close_series.ewm(span=15, adjust=False, min_periods=15).mean()
    ema2 = ema1.ewm(span=15, adjust=False, min_periods=15).mean()
    ema3 = ema2.ewm(span=15, adjust=False, min_periods=15).mean()
    # TRIX = ROC of triple EMA (period=9)
    trix_raw = ema3.pct_change(periods=9) * 100
    trix = trix_raw.values
    
    # Align TRIX to 6h
    trix_aligned = align_htf_to_ltf(prices, df_1d, trix)
    
    # Daily EMA34 for trend filter
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    volume_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_avg * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(trix_aligned[i]) or np.isnan(ema34_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: TRIX crosses above zero + above daily EMA34 + volume confirmation
            if trix_aligned[i] > 0 and trix_aligned[i-1] <= 0 and close[i] > ema34_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: TRIX crosses below zero + below daily EMA34 + volume confirmation
            elif trix_aligned[i] < 0 and trix_aligned[i-1] >= 0 and close[i] < ema34_aligned[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: TRIX crosses below zero OR price below daily EMA34
            if trix_aligned[i] < 0 and trix_aligned[i-1] >= 0 or close[i] < ema34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: TRIX crosses above zero OR price above daily EMA34
            if trix_aligned[i] > 0 and trix_aligned[i-1] <= 0 or close[i] > ema34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals