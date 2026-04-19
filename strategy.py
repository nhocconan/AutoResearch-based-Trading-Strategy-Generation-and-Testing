#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_TRIX_Volume_Spike_Chop_Regime"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for TRIX and Chop calculation
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate TRIX on 1w close (12-period EMA of EMA of EMA)
    close_1w = df_1w['close'].values
    ema1 = pd.Series(close_1w).ewm(span=12, adjust=False).mean().values
    ema2 = pd.Series(ema1).ewm(span=12, adjust=False).mean().values
    ema3 = pd.Series(ema2).ewm(span=12, adjust=False).mean().values
    trix_raw = np.diff(ema3, prepend=ema3[0]) / ema3 * 100
    
    # Calculate Chop on 1w high/low (14-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First TR is just high-low
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    highest_high14 = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
    lowest_low14 = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr14 / (highest_high14 - lowest_low14)) / np.log10(14)
    
    # Align TRIX and Chop to 1d timeframe
    trix = align_htf_to_ltf(prices, df_1w, trix_raw)
    chop_aligned = align_htf_to_ltf(prices, df_1w, chop, additional_delay_bars=0)
    
    # Volume spike on 1d (volume > 2.0 * 20-period average)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 35  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(trix[i]) or np.isnan(chop_aligned[i]):
            signals[i] = 0.0
            continue
            
        # Chop regime: > 61.8 = ranging (mean revert), < 38.2 = trending
        is_ranging = chop_aligned[i] > 61.8
        is_trending = chop_aligned[i] < 38.2
        
        # Volume confirmation
        vol_confirm = volume_spike[i]
        
        if position == 0:
            # Long conditions: TRIX > 0 in ranging OR TRIX rising in trending + volume spike
            if ((is_ranging and trix[i] > 0) or (is_trending and trix[i] > trix[i-1])) and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short conditions: TRIX < 0 in ranging OR TRIX falling in trending + volume spike
            elif ((is_ranging and trix[i] < 0) or (is_trending and trix[i] < trix[i-1])) and vol_confirm:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long exit: TRIX turns negative OR chop shifts to strong trending
            if trix[i] < 0 or chop_aligned[i] < 20:  # Strong trend emerging
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short exit: TRIX turns positive OR chop shifts to strong trending
            if trix[i] > 0 or chop_aligned[i] < 20:  # Strong trend emerging
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals