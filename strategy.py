#!/usr/bin/env python3
"""
Hypothesis: 4h timeframe with 1d Williams %R and volume confirmation. 
Williams %R identifies overbought/oversold conditions on the daily chart.
Enter long when Williams %R crosses above -80 from below (oversold bounce) 
with volume > 1.5x 20-period average. Enter short when Williams %R crosses 
below -20 from above (overbought rejection) with volume confirmation.
Use 1d ADX > 20 to filter for trending markets and avoid ranging whipsaws.
Position sizing: 0.25 for entries, 0 for exits.
Target: 75-200 total trades over 4 years (19-50/year).
Williams %R provides reliable reversal signals in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams %R and ADX
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Williams %R (14)
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close_1d) / (highest_high - lowest_low + 1e-10)
    
    # Calculate 1d ADX (14)
    plus_dm = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    minus_dm = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr3 = np.abs(low_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).sum().values / atr
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).sum().values / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align all to 4h
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume confirmation: 4h volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(williams_r_aligned[i]) or np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Williams %R signals
        wr = williams_r_aligned[i]
        wr_prev = williams_r_aligned[i-1] if i > 0 else -100
        
        # Long signal: Williams %R crosses above -80 from below (oversold bounce)
        long_signal = (wr > -80) and (wr_prev <= -80)
        # Short signal: Williams %R crosses below -20 from above (overbought rejection)
        short_signal = (wr < -20) and (wr_prev >= -20)
        
        if position == 0:
            # Long: oversold bounce + volume confirmation + ADX filter
            if long_signal and volume_confirm[i] and (adx_aligned[i] > 20):
                signals[i] = 0.25
                position = 1
            # Short: overbought rejection + volume confirmation + ADX filter
            elif short_signal and volume_confirm[i] and (adx_aligned[i] > 20):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Williams %R rises above -20 (overbought) or ADX weakens
            if (wr >= -20) or (adx_aligned[i] < 15):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Williams %R falls below -80 (oversold) or ADX weakens
            if (wr <= -80) or (adx_aligned[i] < 15):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_1dWilliamsR_Volume_ADX"
timeframe = "4h"
leverage = 1.0