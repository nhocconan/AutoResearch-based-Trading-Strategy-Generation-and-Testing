#!/usr/bin/env python3
"""
6h_WilliamsVixFix_1dTrend_Filter
Hypothesis: Williams Vix Fix (WVF) identifies market exhaustion and potential reversals. 
In trending markets (determined by 1d EMA34), we take WVF signals in the direction of the trend.
WVF > 0.8 signals extreme fear/greed and potential reversal. Combined with trend filter, 
this captures mean-reversion in strong trends and avoids counter-trend trades in weak markets.
Volume confirmation ensures institutional participation. Works in both bull (buy fear) and 
bear (sell greed) markets. Target: 50-150 total trades over 4 years (12-37/year).
"""

name = "6h_WilliamsVixFix_1dTrend_Filter"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema34_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 34:
        ema34_1d[33] = np.mean(close_1d[:34])
        alpha = 2 / (34 + 1)
        for i in range(34, len(close_1d)):
            ema34_1d[i] = alpha * close_1d[i] + (1 - alpha) * ema34_1d[i-1]
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # 1d volume SMA20 for volume confirmation
    volume_1d = df_1d['volume'].values
    vol_sma20_1d = np.full(len(volume_1d), np.nan)
    if len(volume_1d) >= 20:
        vol_sma20_1d[19] = np.mean(volume_1d[:20])
        for i in range(20, len(volume_1d)):
            vol_sma20_1d[i] = (vol_sma20_1d[i-1] * 19 + volume_1d[i]) / 20
    vol_sma20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_sma20_1d)
    
    # Williams Vix Fix: measures market fear/greed
    # WVF = ((Highest Close in period - Low) / (Highest Close in period)) * 100
    # Highest Close = highest close in lookback period (22 periods default)
    lookback = 22
    highest_close = np.full(n, np.nan)
    for i in range(lookback - 1, n):
        highest_close[i] = np.max(close[i - lookback + 1:i + 1])
    
    wvf = np.full(n, np.nan)
    for i in range(lookback - 1, n):
        if not np.isnan(highest_close[i]) and highest_close[i] > 0:
            wvf[i] = ((highest_close[i] - low[i]) / highest_close[i]) * 100
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(lookback - 1, 34)
    
    for i in range(start_idx, n):
        if np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_sma20_1d_aligned[i]) or np.isnan(wvf[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current 6h volume > 1.5x average 1d volume (scaled to 6h)
        vol_6h_approx = vol_sma20_1d_aligned[i] / 4.0
        volume_confirm = volume[i] > 1.5 * vol_6h_approx
        
        # WVF threshold for extreme readings
        wvf_extreme = wvf[i] > 80  # Typically >80 indicates extreme fear/greed
        
        if position == 0:
            # Long: Extreme fear (high WVF) in uptrend with volume
            if wvf_extreme and close[i] > ema34_1d_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Extreme greed (high WVF) in downtrend with volume
            elif wvf_extreme and close[i] < ema34_1d_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Trend reversal or WVF normalization
            if close[i] < ema34_1d_aligned[i] or wvf[i] < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Trend reversal or WVF normalization
            if close[i] > ema34_1d_aligned[i] or wvf[i] < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals