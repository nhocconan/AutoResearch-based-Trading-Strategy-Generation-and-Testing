#!/usr/bin/env python3
"""
12h_1d_trix_volume_trend
Hypothesis: 12-hour strategy using TRIX momentum on daily timeframe for trend direction, 
with volume confirmation and 12-hour price action for entries. TRIX filters noise and 
identifies sustained momentum, working in both bull and bear markets by capturing 
medium-term trends. Volume confirmation avoids false signals. Target: 15-30 trades/year 
(60-120 total over 4 years) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for TRIX trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate TRIX: triple EMA of 15-period
    ema1 = pd.Series(close_1d).ewm(span=15, adjust=False, min_periods=15).mean()
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean()
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean()
    
    # TRIX = (EMA3 - previous EMA3) / previous EMA3 * 100
    trix_raw = (ema3 - ema3.shift(1)) / ema3.shift(1) * 100
    trix = trix_raw.fillna(0).values
    
    # Align TRIX to 12h timeframe (wait for daily close)
    trix_aligned = align_htf_to_ltf(prices, df_1d, trix)
    
    # Volume confirmation: volume > 1.5x 30-period average
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if data not ready
        if np.isnan(trix_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Long entry: positive TRIX (bullish momentum) AND price above prior close with volume
        if (trix_aligned[i] > 0 and close[i] > close[i-1] and vol_confirm[i] and position != 1):
            position = 1
            signals[i] = 0.25
        # Short entry: negative TRIX (bearish momentum) AND price below prior close with volume
        elif (trix_aligned[i] < 0 and close[i] < close[i-1] and vol_confirm[i] and position != -1):
            position = -1
            signals[i] = -0.25
        # Exit: TRIX crosses zero or price reverses against position
        elif position == 1 and trix_aligned[i] < 0:
            position = 0
            signals[i] = 0.0
        elif position == -1 and trix_aligned[i] > 0:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_1d_trix_volume_trend"
timeframe = "12h"
leverage = 1.0