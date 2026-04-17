#!/usr/bin/env python3
"""
Hypothesis: 4h timeframe with 1d ATR-based volatility breakout and volume confirmation.
Go long when price breaks above high + 0.5*ATR(20) with volume > 1.5x average.
Go short when price breaks below low - 0.5*ATR(20) with volume > 1.5x average.
Use 1d ADX > 20 to filter for sufficient momentum and avoid choppy markets.
Position sizing: 0.25 for entries, 0 for exits.
Target: 75-200 total trades over 4 years (19-50/year).
ATR breakouts capture expansion moves that work in both bull and bear markets.
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
    
    # Get 1d data for ATR, breakout levels, and ADX
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d ATR (20)
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr3 = np.abs(low_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d breakout levels: high +/- 0.5*ATR, low +/- 0.5*ATR
    upper_breakout = high_1d + 0.5 * atr
    lower_breakout = low_1d - 0.5 * atr
    
    # Calculate 1d ADX (14)
    plus_dm = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    minus_dm = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    
    tr_adx = np.maximum(tr1, np.maximum(tr2, tr3))  # reuse TR
    
    atr_adx = pd.Series(tr_adx).rolling(window=14, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).sum().values / atr_adx
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).sum().values / atr_adx
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Volume filter: 1.5x 20-period average
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all to 4h
    upper_breakout_aligned = align_htf_to_ltf(prices, df_1d, upper_breakout)
    lower_breakout_aligned = align_htf_to_ltf(prices, df_1d, lower_breakout)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper_breakout_aligned[i]) or np.isnan(lower_breakout_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend strength from ADX
        strong_momentum = adx_aligned[i] > 20
        
        if position == 0:
            # Long: price breaks above upper level, volume spike, sufficient momentum
            if (close[i] > upper_breakout_aligned[i] and 
                volume[i] > vol_ma_20_aligned[i] * 1.5 and 
                strong_momentum):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower level, volume spike, sufficient momentum
            elif (close[i] < lower_breakout_aligned[i] and 
                  volume[i] > vol_ma_20_aligned[i] * 1.5 and 
                  strong_momentum):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns below lower breakout or momentum weakens
            if close[i] < lower_breakout_aligned[i] or adx_aligned[i] < 15:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns above upper breakout or momentum weakens
            if close[i] > upper_breakout_aligned[i] or adx_aligned[i] < 15:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_1dATRBreakout_Volume_ADX"
timeframe = "4h"
leverage = 1.0