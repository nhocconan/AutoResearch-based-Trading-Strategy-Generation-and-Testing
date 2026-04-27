#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h TRIX + volume spike + choppiness regime filter
# TRIX (1-period ROC of EMA) filters noise and identifies momentum.
# In trending markets, TRIX crosses above/below zero signal momentum shifts.
# Volume spike confirms breakout conviction. Choppiness regime filter avoids ranging markets.
# Works in bull/bear via momentum + volume confirmation.
# Target: 75-200 total trades over 4 years (~19-50/year) to avoid fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for choppiness regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # TRIX: 1-period ROC of triple EMA (15-period)
    ema1 = pd.Series(close).ewm(span=15, adjust=False, min_periods=15).mean()
    ema2 = ema1.ewm(span=15, adjust=False, min_periods=15).mean()
    ema3 = ema2.ewm(span=15, adjust=False, min_periods=15).mean()
    trix = 100 * (ema3.pct_change(1))
    trix = trix.values
    
    # Choppiness Index (14-period)
    def true_range(high, low, close_prev):
        return np.maximum(high - low, np.maximum(np.abs(high - close_prev), np.abs(low - close_prev)))
    
    tr1 = true_range(high_1d, low_1d, np.concatenate([[close_1d[0]], close_1d[:-1]]))
    atr1 = pd.Series(tr1).rolling(window=14, min_periods=14).mean().values
    
    hh = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    chop = 100 * np.log10(atr1 * 14 / (hh - ll)) / np.log10(14)
    
    # Align indicators to 4h timeframe
    trix_aligned = align_htf_to_ltf(prices, prices, trix)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Volume filter: volume > 2.0 x 20-period average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need TRIX (15*3=45), chop (14), volume MA (20)
    start_idx = max(45, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(trix_aligned[i]) or np.isnan(chop_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_now = volume[i]
        vol_avg = vol_ma_20[i]
        
        # Volume filter: significant volume spike
        vol_filter = vol_now > 2.0 * vol_avg
        
        # Momentum from TRIX
        trix_now = trix_aligned[i]
        trix_prev = trix_aligned[i-1]
        
        # Choppiness regime: chop > 61.8 = ranging (avoid), chop < 38.2 = trending
        chop_now = chop_aligned[i]
        trending_regime = chop_now < 38.2
        
        if position == 0:
            # Long: TRIX crosses above zero + volume + trending regime
            if trix_prev <= 0 and trix_now > 0 and vol_filter and trending_regime:
                signals[i] = size
                position = 1
            # Short: TRIX crosses below zero + volume + trending regime
            elif trix_prev >= 0 and trix_now < 0 and vol_filter and trending_regime:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: TRIX crosses below zero
            if trix_now < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: TRIX crosses above zero
            if trix_now > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_TRIX_Volume_Chop"
timeframe = "4h"
leverage = 1.0