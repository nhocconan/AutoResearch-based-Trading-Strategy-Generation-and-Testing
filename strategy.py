#!/usr/bin/env python3
# Hypothesis: 12h TRIX with volume spike and 1-day Choppiness index filter for regime detection
# Long when TRIX crosses above zero, volume > 2x 20-period average, and 1-day Choppiness > 61.8 (ranging)
# Short when TRIX crosses below zero, volume > 2x 20-period average, and 1-day Choppiness > 61.8 (ranging)
# Exit when TRIX crosses back through zero or Choppiness drops below 38.2 (trending)
# Position size: 0.25 to limit drawdown and reduce trade frequency

name = "12h_Trix_Volume_Chop"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 12-period TRIX (triple smoothed EMA of ROC)
    # ROC = (close - close.shift(12)) / close.shift(12)
    roc = np.zeros_like(close)
    roc[12:] = (close[12:] - close[:-12]) / close[:-12]
    
    # Triple EMA smoothing
    ema1 = pd.Series(roc).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema2 = pd.Series(ema1).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema3 = pd.Series(ema2).ewm(span=12, adjust=False, min_periods=12).mean().values
    trix = ema3 * 100  # Scale for readability
    
    # Volume spike: current volume > 2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_spike = volume > (2.0 * vol_ma.values)
    
    # Get 1d data for Choppiness index
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate 1-day Choppiness index (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    
    # ATR(14) - sum of TR over 14 periods
    atr = np.zeros_like(tr)
    for i in range(14, len(tr)):
        atr[i] = np.nansum(tr[i-13:i+1])  # Simple sum for ATR
    
    # Highest high and lowest low over 14 periods
    hh = np.zeros_like(high_1d)
    ll = np.zeros_like(low_1d)
    for i in range(14, len(high_1d)):
        hh[i] = np.max(high_1d[i-13:i+1])
        ll[i] = np.min(low_1d[i-13:i+1])
    
    # Chop = 100 * log10( sum(TR14) / (HH14 - LL14) ) / log10(14)
    chop = np.full_like(close_1d, np.nan)
    for i in range(14, len(close_1d)):
        if hh[i] != ll[i]:  # Avoid division by zero
            chop[i] = 100 * np.log10(atr[i] / (hh[i] - ll[i])) / np.log10(14)
    
    # Align Choppiness to 12h timeframe (waits for daily close)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Need enough data for TRIX and Chop
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(trix[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: TRIX crosses above zero, volume spike, chop > 61.8 (ranging)
            if (trix[i] > 0 and trix[i-1] <= 0 and 
                vol_spike[i] and chop_aligned[i] > 61.8):
                signals[i] = 0.25
                position = 1
            # Enter short: TRIX crosses below zero, volume spike, chop > 61.8 (ranging)
            elif (trix[i] < 0 and trix[i-1] >= 0 and 
                  vol_spike[i] and chop_aligned[i] > 61.8):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: TRIX crosses below zero OR chop < 38.2 (trending)
            if (trix[i] < 0 and trix[i-1] >= 0) or chop_aligned[i] < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: TRIX crosses above zero OR chop < 38.2 (trending)
            if (trix[i] > 0 and trix[i-1] <= 0) or chop_aligned[i] < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals