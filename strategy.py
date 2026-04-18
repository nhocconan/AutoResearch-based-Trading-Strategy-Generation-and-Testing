#!/usr/bin/env python3
"""
6h_RSI_Trend_Signal_1dATRFilter_v1
Hypothesis: Use RSI(14) on 6h to detect momentum extremes in trending markets, filtered by 1d ATR to confirm volatility regime. 
Long when RSI crosses above 40 (bullish momentum) and 1d ATR is expanding (volatility > 50-day median). 
Short when RSI crosses below 60 (bearish momentum) and 1d ATR is expanding. 
Exit when RSI returns to neutral zone (40-60) or ATR contracts. 
ATR filter prevents trading in low-volatility chop, reducing false signals. 
Designed for 6-12 trades per year per symbol, targeting 50-100 total over 4 years.
Works in bull/bear by following momentum with volatility filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for ATR filter
    df_1d = get_htf_data(prices, '1d')
    
    # 1d ATR(14)
    atr_period = 14
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    atr_1d = np.full_like(close_1d, np.nan)
    
    if len(close_1d) >= atr_period:
        tr1 = high_1d[1:] - low_1d[1:]
        tr2 = np.abs(high_1d[1:] - close_1d[:-1])
        tr3 = np.abs(low_1d[1:] - close_1d[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])  # align with close_1d
        
        # Wilder smoothing
        atr_1d[atr_period-1] = np.nanmean(tr[1:atr_period+1])
        for i in range(atr_period, len(close_1d)):
            atr_1d[i] = (atr_1d[i-1] * (atr_period - 1) + tr[i]) / atr_period
    
    # 1d ATR 50-day median for regime filter
    atr_median_50 = np.full_like(close_1d, np.nan)
    if len(close_1d) >= 50:
        for i in range(49, len(close_1d)):
            atr_median_50[i] = np.nanmedian(atr_1d[i-49:i+1])
    
    # Align 1d ATR and median to 6h
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    atr_median_50_aligned = align_htf_to_ltf(prices, df_1d, atr_median_50)
    
    # 6h RSI(14)
    rsi_period = 14
    rsi = np.full_like(close, np.nan)
    
    if len(close) >= rsi_period + 1:
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.full_like(close, np.nan)
        avg_loss = np.full_like(close, np.nan)
        
        # First average
        avg_gain[rsi_period] = np.mean(gain[:rsi_period])
        avg_loss[rsi_period] = np.mean(loss[:rsi_period])
        
        # Wilder smoothing
        for i in range(rsi_period + 1, len(close)):
            avg_gain[i] = (avg_gain[i-1] * (rsi_period - 1) + gain[i-1]) / rsi_period
            avg_loss[i] = (avg_loss[i-1] * (rsi_period - 1) + loss[i-1]) / rsi_period
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(rsi_period + 1, atr_period + 49)  # RSI needs 15, ATR median needs 50
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(rsi[i]) or np.isnan(atr_1d_aligned[i]) or 
            np.isnan(atr_median_50_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: ATR > 50-day median (expanding volatility)
        vol_expanding = atr_1d_aligned[i] > atr_median_50_aligned[i]
        
        if position == 0:
            # Long: RSI crosses above 40 + volatility expanding
            if i > 0 and not np.isnan(rsi[i-1]) and rsi[i-1] <= 40 and rsi[i] > 40 and vol_expanding:
                signals[i] = 0.25
                position = 1
            # Short: RSI crosses below 60 + volatility expanding
            elif i > 0 and not np.isnan(rsi[i-1]) and rsi[i-1] >= 60 and rsi[i] < 60 and vol_expanding:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: RSI returns to neutral (<40) or volatility contracts
            if rsi[i] < 40 or not vol_expanding:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: RSI returns to neutral (>60) or volatility contracts
            if rsi[i] > 60 or not vol_expanding:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_RSI_Trend_Signal_1dATRFilter_v1"
timeframe = "6h"
leverage = 1.0