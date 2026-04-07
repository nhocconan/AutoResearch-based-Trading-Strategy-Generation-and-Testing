#!/usr/bin/env python3
"""
4h_trix_volume_regime_v1
Hypothesis: TRIX (Triple Exponential Average) on 4h combined with volume confirmation and 
choppiness regime filter works on 4h timeframe. TRIX filters out noise and identifies 
momentum shifts. Volume confirms momentum strength. Choppiness filter avoids ranging markets.
Targets 20-50 trades/year (80-200 over 4 years). Works in both bull and bear markets 
by adapting to regime via choppiness filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_trix_volume_regime_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily data for choppiness regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate TRIX on 4h (15-period)
    ema1 = pd.Series(close).ewm(span=15, adjust=False).mean()
    ema2 = ema1.ewm(span=15, adjust=False).mean()
    ema3 = ema2.ewm(span=15, adjust=False).mean()
    trix = 100 * (ema3.pct_change())
    trix_values = trix.values
    
    # Calculate 14-period choppiness index on daily
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # ATR14
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Sum of True Range over 14 periods
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Max and min close over 14 periods
    max_h = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_l = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index: 100 * log10(tr_sum / (max_h - min_l)) / log10(14)
    range_14 = max_h - min_l
    # Avoid division by zero
    range_14 = np.where(range_14 == 0, 1e-10, range_14)
    choppiness = 100 * np.log10(tr_sum / range_14) / np.log10(14)
    
    # Align daily choppiness to 4h timeframe
    chop_4h = align_htf_to_ltf(prices, df_1d, choppiness)
    
    # 20-period volume average on 4h
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(trix_values[i]) or np.isnan(chop_4h[i]) or 
            np.isnan(vol_sma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3x average volume
        vol_confirm = volume[i] > 1.3 * vol_sma[i]
        
        # Choppiness regime: < 38.2 = trending, > 61.8 = ranging
        is_trending = chop_4h[i] < 38.2
        
        if position == 1:  # Long position
            # Exit: TRIX turns negative OR choppy market
            if trix_values[i] < 0 or not is_trending:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: TRIX turns positive OR choppy market
            if trix_values[i] > 0 or not is_trending:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # TRIX long signal: TRIX crosses above zero in trending market
            if (trix_values[i] > 0 and 
                trix_values[i-1] <= 0 and 
                vol_confirm and 
                is_trending):
                position = 1
                signals[i] = 0.25
            # TRIX short signal: TRIX crosses below zero in trending market
            elif (trix_values[i] < 0 and 
                  trix_values[i-1] >= 0 and 
                  vol_confirm and 
                  is_trending):
                position = -1
                signals[i] = -0.25
    
    return signals