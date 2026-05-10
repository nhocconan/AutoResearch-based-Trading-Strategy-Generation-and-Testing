#!/usr/bin/env python3
# 12h_Phase_Accumulation_Strategy
# Hypothesis: Uses 1-week ROC to detect long-term momentum and 1-day Williams %R for oversold/overbought conditions.
# In bull markets (weekly ROC > 0), we go long when daily Williams %R < -80 (oversold) with volume confirmation.
# In bear markets (weekly ROC < 0), we go short when daily Williams %R > -20 (overbought) with volume confirmation.
# Uses 12h price action for entry timing and volatility filter to avoid choppy markets.
# Designed for very low trade frequency (<15/year) to minimize fee drag.

name = "12h_Phase_Accumulation_Strategy"
timeframe = "12h"
leverage = 1.0

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
    
    # Get 1-week data for long-term momentum
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    # Weekly ROC(12) for trend filter (3-month momentum)
    roc_12_1w = np.full_like(close_1w, np.nan)
    if len(close_1w) >= 13:
        roc_12_1w[12:] = (close_1w[12:] - close_1w[:-12]) / close_1w[:-12] * 100
    roc_12_1w_aligned = align_htf_to_ltf(prices, df_1w, roc_12_1w)
    
    # Get 1-day data for oversold/overbought signals
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    # Williams %R(14) on daily data
    highest_high = np.full_like(high_1d, np.nan)
    lowest_low = np.full_like(low_1d, np.nan)
    for i in range(13, len(high_1d)):
        highest_high[i] = np.max(high_1d[i-13:i+1])
        lowest_low[i] = np.min(low_1d[i-13:i+1])
    wr_14_1d = np.full_like(close_1d, np.nan)
    for i in range(13, len(close_1d)):
        if highest_high[i] > lowest_low[i]:
            wr_14_1d[i] = (highest_high[i] - close_1d[i]) / (highest_high[i] - lowest_low[i]) * -100
        else:
            wr_14_1d[i] = -50  # neutral when range is zero
    wr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, wr_14_1d)
    
    # Volatility filter: 12h ATR(24) to avoid choppy markets
    def true_range(h, l, c_prev):
        tr1 = h - l
        tr2 = np.abs(h - c_prev)
        tr3 = np.abs(l - c_prev)
        return np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_raw = np.full_like(close, np.nan)
    for i in range(1, n):
        atr_raw[i] = true_range(high[i], low[i], close[i-1])
    
    atr_24 = np.full_like(close, np.nan)
    if len(atr_raw) >= 24:
        for i in range(23, n):
            atr_24[i] = np.mean(atr_raw[i-23:i+1])
    
    # Volume confirmation: 24-period average (~12 days)
    def mean_arr(arr, p):
        res = np.full_like(arr, np.nan)
        if len(arr) >= p:
            for i in range(p-1, len(arr)):
                res[i] = np.mean(arr[i-p+1:i+1])
        return res
    vol_ma = mean_arr(volume, 24)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 24)  # need enough history for calculations
    
    for i in range(start_idx, n):
        if np.isnan(roc_12_1w_aligned[i]) or np.isnan(wr_14_1d_aligned[i]) or \
           np.isnan(atr_24[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.5x average
        volume_confirm = volume[i] > 1.5 * vol_ma[i] if vol_ma[i] > 0 else False
        
        # Volatility filter: avoid extremely high volatility (chaotic markets)
        vol_filter = atr_24[i] < np.mean(atr_24[max(0, i-100):i+1]) * 2 if i >= 100 else True
        
        if position == 0:
            # Long: bullish long-term momentum AND oversold conditions with volume
            if roc_12_1w_aligned[i] > 0 and wr_14_1d_aligned[i] < -80 and \
               volume_confirm and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: bearish long-term momentum AND overbought conditions with volume
            elif roc_12_1w_aligned[i] < 0 and wr_14_1d_aligned[i] > -20 and \
                 volume_confirm and vol_filter:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: momentum turns negative OR overbought conditions
            if roc_12_1w_aligned[i] <= 0 or wr_14_1d_aligned[i] > -20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: momentum turns positive OR oversold conditions
            if roc_12_1w_aligned[i] >= 0 or wr_14_1d_aligned[i] < -80:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals