#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h TRIX momentum with volume spike and 1d volatility filter.
# TRIX (12) crosses above signal line (9) + volume > 1.5x average + 1d ATR < 0.03*price (low volatility regime) = long.
# TRIX crosses below signal line + volume spike + low volatility = short.
# Exit when TRIX crosses back through zero line.
# Uses 1d ATR for regime filter to avoid high-volatility chop.
# Target: 20-40 trades/year to minimize fee dash while capturing momentum bursts.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for ATR volatility filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1-day ATR(14) for volatility regime
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - close_1d[:-1]))
    tr2 = np.maximum(np.abs(low_1d[1:] - close_1d[:-1]), tr1)
    tr = np.concatenate([[np.inf], tr2])  # first TR undefined
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_pct = atr_14 / close_1d  # ATR as percentage of price
    
    # Align ATR percentage to 4h timeframe
    atr_14_pct_aligned = align_htf_to_ltf(prices, df_1d, atr_14_pct)
    
    # Get 4h data for TRIX calculation
    # TRIX = triple EMA of ROC, then signal line = EMA of TRIX
    # ROC(1) = (close/t-1 - close/t-2)/close/t-2
    roc = np.diff(close) / close[:-1]
    roc = np.concatenate([[0], roc])  # first ROC undefined
    
    # Triple EMA smoothing
    ema1 = pd.Series(roc).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema2 = pd.Series(ema1).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema3 = pd.Series(ema2).ewm(span=12, adjust=False, min_periods=12).mean().values
    trix = ema3 * 100  # scale for readability
    
    # Signal line = EMA of TRIX
    signal_line = pd.Series(trix).ewm(span=9, adjust=False, min_periods=9).mean().values
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(trix[i]) or np.isnan(signal_line[i]) or 
            np.isnan(atr_14_pct_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Low volatility regime: ATR < 3% of price
        low_vol = atr_14_pct_aligned[i] < 0.03
        
        # Long condition: TRIX crosses above signal line, volume spike, low volatility
        if (trix[i] > signal_line[i] and trix[i-1] <= signal_line[i-1] and
            volume_filter[i] and low_vol):
            signals[i] = 0.25
            position = 1
        # Short condition: TRIX crosses below signal line, volume spike, low volatility
        elif (trix[i] < signal_line[i] and trix[i-1] >= signal_line[i-1] and
              volume_filter[i] and low_vol):
            signals[i] = -0.25
            position = -1
        # Exit conditions: TRIX crosses zero line
        elif position == 1 and trix[i] < 0:
            signals[i] = 0.0
            position = 0
        elif position == -1 and trix[i] > 0:
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_TRIX_Volume_LowVol_Filter"
timeframe = "4h"
leverage = 1.0