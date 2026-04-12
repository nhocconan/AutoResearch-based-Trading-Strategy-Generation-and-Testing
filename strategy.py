#!/usr/bin/env python3
"""
12h_1d_Premium_Reversion
Hypothesis: On 12h timeframe, enter long when price crosses below daily VWAP with volume contraction and 1d ATR contraction (low volatility regime), exit on VWAP reversion. Short when price crosses above daily VWAP with volume expansion and ATR expansion (high volatility regime), exit on VWAP reversion. Uses daily VWAP as mean, ATR regime filter, and volume change filter. Designed for mean reversion in low volatility and momentum in high volatility regimes. Targets 15-30 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_Premium_Reversion"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === DAILY VWAP ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Typical price
    tp_1d = (high_1d + low_1d + close_1d) / 3
    # VWAP = sum(tp * volume) / sum(volume)
    vwap_num = np.cumsum(tp_1d * volume_1d)
    vwap_den = np.cumsum(volume_1d)
    vwap_1d = np.where(vwap_den != 0, vwap_num / vwap_den, tp_1d)
    
    # Align VWAP to 12h
    vwap_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    
    # === DAILY ATR(14) FOR VOLATILITY REGIME ===
    # True Range
    tr = np.maximum(high_1d - low_1d, 
                    np.maximum(np.abs(high_1d - np.roll(close_1d, 1)), 
                               np.abs(low_1d - np.roll(close_1d, 1))))
    tr[0] = high_1d[0] - low_1d[0]
    
    # ATR(14)
    atr_1d = np.full_like(tr, np.nan)
    if len(tr) >= 14:
        atr_1d[13] = np.mean(tr[0:14])
        for i in range(14, len(tr)):
            atr_1d[i] = (atr_1d[i-1] * 13 + tr[i]) / 14
    
    # Align ATR to 12h
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # ATR percentile rank (20-period) for regime
    atr_rank = np.full_like(atr_aligned, np.nan)
    for i in range(20, len(atr_aligned)):
        if not np.isnan(atr_aligned[i]):
            window = atr_aligned[max(0, i-19):i+1]
            valid = window[~np.isnan(window)]
            if len(valid) > 0:
                rank = np.sum(valid <= atr_aligned[i]) / len(valid) * 100
                atr_rank[i] = rank
    
    # === VOLUME CHANGE (12h) ===
    vol_change = np.zeros_like(volume)
    vol_ma_5 = np.zeros_like(volume)
    if len(volume) >= 5:
        vol_ma_5[4] = np.mean(volume[0:5])
        for i in range(5, len(volume)):
            vol_ma_5[i] = (vol_ma_5[i-1] * 4 + volume[i]) / 5
    
    vol_change = volume / vol_ma_5 - 1  # percentage change from 5-period MA
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # start after warmup
        # Skip if indicators not available
        if (np.isnan(vwap_aligned[i]) or np.isnan(atr_aligned[i]) or 
            np.isnan(atr_rank[i]) or np.isnan(vol_change[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Long: price below VWAP, low volatility regime (ATR rank < 30), volume contraction
        long_entry = (close[i] < vwap_aligned[i] and 
                      atr_rank[i] < 30 and 
                      vol_change[i] < -0.1)  # volume below 5MA
        
        # Short: price above VWAP, high volatility regime (ATR rank > 70), volume expansion
        short_entry = (close[i] > vwap_aligned[i] and 
                       atr_rank[i] > 70 and 
                       vol_change[i] > 0.2)  # volume above 5MA
        
        # Exit: price crosses VWAP (mean reversion)
        exit_long = close[i] > vwap_aligned[i]
        exit_short = close[i] < vwap_aligned[i]
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals