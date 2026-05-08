#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h TRIX with 12h trend filter and volume confirmation
# TRIX (triple-smoothed EMA) filters noise and identifies momentum.
# We go long when TRIX crosses above zero and rising, short when crosses below zero and falling,
# aligned with 12h EMA(50) trend and confirmed by volume spike.
# Designed for low trade frequency in both bull and bear markets.
# Target: 50-150 total trades over 4 years = 12-37/year

name = "4h_TRIX_12hTrend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data once
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA(50) for trend direction
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Calculate TRIX on 4h data (triple EMA of 1-period ROC)
    # TRIX = EMA(EMA(EMA(ROC, n), n), n) where ROC = (close/close.prev - 1) * 100
    roc = np.diff(close, prepend=close[0]) / close * 100
    ema1 = pd.Series(roc).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean().values
    trix = ema3
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema50_12h_aligned[i]) or np.isnan(trix[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema50_12h_val = ema50_12h_aligned[i]
        trix_val = trix[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: TRIX crosses above zero and rising + uptrend + volume spike
            if i > 0:
                trix_prev = trix[i-1]
                trix_rising = trix_val > trix_prev
                if (trix_val > 0 and trix_prev <= 0 and trix_rising and 
                    close[i] > ema50_12h_val and 
                    vol_spike):
                    signals[i] = 0.25
                    position = 1
            # Enter short: TRIX crosses below zero and falling + downtrend + volume spike
            if i > 0:
                trix_prev = trix[i-1]
                trix_falling = trix_val < trix_prev
                if (trix_val < 0 and trix_prev >= 0 and trix_falling and 
                    close[i] < ema50_12h_val and 
                    vol_spike):
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit long: TRIX crosses below zero OR price breaks below trend
            if i > 0:
                trix_prev = trix[i-1]
                if (trix_val < 0 and trix_prev >= 0) or close[i] < ema50_12h_val:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
        elif position == -1:
            # Exit short: TRIX crosses above zero OR price breaks above trend
            if i > 0:
                trix_prev = trix[i-1]
                if (trix_val > 0 and trix_prev <= 0) or close[i] > ema50_12h_val:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals