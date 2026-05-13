#!/usr/bin/env python3
"""
4h_TRIX_Signal_ZeroCross_12hTrend_Volume
Hypothesis: TRIX (Triple Exponential Moving Average) zero-cross signals capture momentum shifts.
When combined with 12h EMA trend filter and volume confirmation, this creates a robust
trend-following strategy that works in both bull and bear markets by only taking trades
in the direction of the higher timeframe trend. Target: 20-40 trades/year to minimize
fee drag while maintaining edge.
"""

name = "4h_TRIX_Signal_ZeroCross_12hTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend filter (once before loop)
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate TRIX on 4h close (15-period standard)
    # TRIX = EMA(EMA(EMA(close, 15), 15), 15)
    ema1 = pd.Series(close).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean().values
    trix = pd.Series(ema3).pct_change() * 100  # Percentage change
    
    # Calculate 12h EMA50 for trend filter
    ema50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        if position == 0:
            # LONG: TRIX crosses above zero with volume and 12h uptrend
            if (trix[i] > 0 and trix[i-1] <= 0 and 
                volume_confirm[i] and 
                close[i] > trend_12h_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: TRIX crosses below zero with volume and 12h downtrend
            elif (trix[i] < 0 and trix[i-1] >= 0 and 
                  volume_confirm[i] and 
                  close[i] < trend_12h_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: TRIX crosses below zero or trend turns down
            if (trix[i] < 0 and trix[i-1] >= 0) or \
               (close[i] < trend_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: TRIX crosses above zero or trend turns up
            if (trix[i] > 0 and trix[i-1] <= 0) or \
               (close[i] > trend_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals