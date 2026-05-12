#!/usr/bin/env python3
"""
6H_TRIX_ZERO_CROSS_1DVOLUME_REGIME
Hypothesis: Use TRIX (triple smoothed ROC) zero crosses as momentum signals, filtered by daily volume regime and 1d trend.
- Long when TRIX crosses above zero AND volume > 1.5x 20-day average AND price > 50-day EMA
- Short when TRIX crosses below zero AND volume > 1.5x 20-day average AND price < 50-day EMA
- Exit when TRIX crosses back through zero or volatility regime changes
- Volume regime filter ensures trades occur during institutional participation
- Trend filter (50-day EMA) aligns with higher timeframe direction to reduce whipsaw
- Target: 25-35 trades/year (100-140 total over 4 years) within 6h limits
- Works in bull markets (TRIX catches momentum) and bear markets (short signals from downside crosses)
"""
name = "6H_TRIX_ZERO_CROSS_1DVOLUME_REGIME"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume regime: volume > 1.5x 20-period average (using close price for ROC calculation)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_regime = volume > (1.5 * vol_ma)
    
    # 1d data for TRIX calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # TRIX calculation: triple smoothed 1-period ROC
    # ROC = (close - close.shift(1)) / close.shift(1) * 100
    roc = pd.Series(df_1d['close']).pct_change() * 100
    # Triple exponential smoothing
    ema1 = roc.ewm(span=15, adjust=False, min_periods=15).mean()
    ema2 = ema1.ewm(span=15, adjust=False, min_periods=15).mean()
    ema3 = ema2.ewm(span=15, adjust=False, min_periods=15).mean()
    trix = ema3.values
    
    # Align TRIX to 6h timeframe
    trix_aligned = align_htf_to_ltf(prices, df_1d, trix)
    
    # 50-day EMA for trend filter
    ema_50 = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(15, n):  # Start after TRIX warmup
        if (np.isnan(trix_aligned[i]) or 
            np.isnan(trix_aligned[i-1]) or 
            np.isnan(ema_50_aligned[i]) or 
            np.isnan(volume_regime[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: TRIX crosses above zero + volume regime + price above 50-day EMA
            if (trix_aligned[i-1] <= 0 and trix_aligned[i] > 0 and 
                volume_regime[i] and 
                close[i] > ema_50_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: TRIX crosses below zero + volume regime + price below 50-day EMA
            elif (trix_aligned[i-1] >= 0 and trix_aligned[i] < 0 and 
                  volume_regime[i] and 
                  close[i] < ema_50_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: TRIX crosses back below zero OR volatility regime ends
            if trix_aligned[i] < 0 or not volume_regime[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: TRIX crosses back above zero OR volatility regime ends
            if trix_aligned[i] > 0 or not volume_regime[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals