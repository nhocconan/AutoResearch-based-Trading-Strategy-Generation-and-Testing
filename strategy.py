#!/usr/bin/env python3
"""
4h_TRIX_VolumeSpike_CM_Signal
TRIX momentum with volume spike confirmation and Chande Momentum Oscillator filter:
- Long when TRIX crosses above zero + volume spike + CMO > 0
- Short when TRIX crosses below zero + volume spike + CMO < 0
- Exit when TRIX crosses back opposite direction
- Uses 1d trend filter: price above/below 50-period EMA
- Designed for ~30-50 trades/year per symbol
Works in bull (captures momentum) and bear (mean reversion via CMO filter) markets
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def ema(src, length):
    """Exponential Moving Average"""
    return pd.Series(src).ewm(span=length, adjust=False, min_periods=length).mean()

def trix(close, length=12):
    """TRIX indicator"""
    ema1 = ema(close, length)
    ema2 = ema(ema1, length)
    ema3 = ema(ema2, length)
    return pd.Series(ema3).pct_change() * 100

def chande_momentum_oscillator(close, length=14):
    """Chande Momentum Oscillator"""
    mom = pd.Series(close).diff()
    pos_mom = mom.copy()
    neg_mom = mom.copy()
    pos_mom[pos_mom < 0] = 0
    neg_mom[neg_mom > 0] = 0
    sum_pos = pd.Series(pos_mom).rolling(window=length, min_periods=length).sum()
    sum_neg = pd.Series(np.abs(neg_mom)).rolling(window=length, min_periods=length).sum()
    cmo = 100 * (sum_pos - sum_neg) / (sum_pos + sum_neg)
    return cmo.values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 for trend filter
    ema50_1d = ema(close_1d, 50)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate TRIX on 4h
    trix_vals = trix(close, 12)
    
    # Calculate volume spike (volume > 1.5 * 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_spike = volume > (1.5 * vol_ma.values)
    
    # Calculate CMO
    cmo = chande_momentum_oscillator(close, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # need sufficient data for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(trix_vals[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(cmo[i])):
            signals[i] = 0.0
            continue
        
        # TRIX zero cross
        trix_cross_above = trix_vals[i] > 0 and trix_vals[i-1] <= 0
        trix_cross_below = trix_vals[i] < 0 and trix_vals[i-1] >= 0
        
        # 1d trend filter
        uptrend = close[i] > ema50_1d_aligned[i]
        downtrend = close[i] < ema50_1d_aligned[i]
        
        if position == 0:
            # Long: TRIX cross above zero + volume spike + CMO > 0 + uptrend
            if trix_cross_above and vol_spike[i] and cmo[i] > 0 and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: TRIX cross below zero + volume spike + CMO < 0 + downtrend
            elif trix_cross_below and vol_spike[i] and cmo[i] < 0 and downtrend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: TRIX cross below zero
            if trix_cross_below:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: TRIX cross above zero
            if trix_cross_above:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_TRIX_VolumeSpike_CM_Signal"
timeframe = "4h"
leverage = 1.0