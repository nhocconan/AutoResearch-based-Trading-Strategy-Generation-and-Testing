#!/usr/bin/env python3
"""
6h_WilliamsVixFix_MeanReversion_1dTrend
Hypothesis: Williams Vix Fix (WVF) identifies volatility spikes and mean reversion opportunities.
In bull markets: buy WVF > 0.8 (panic) when price > 1d EMA50 (uptrend).
In bear markets: sell short WVF > 0.8 when price < 1d EMA50 (downtrend).
Uses volume confirmation to avoid false signals. Target: 50-150 trades over 4 years.
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
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate EMA(50) on 1d close
    close_1d = df_1d['close'].values
    ema_period = 50
    ema_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= ema_period:
        ema_1d[ema_period-1] = np.mean(close_1d[:ema_period])
        multiplier = 2 / (ema_period + 1)
        for i in range(ema_period, len(close_1d)):
            ema_1d[i] = (close_1d[i] * multiplier) + (ema_1d[i-1] * (1 - multiplier))
    
    # Align 1d EMA to 6h timeframe
    ema_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate Williams Vix Fix (WVF) on 6h data
    # WVF = ((Highest High in period - Low) / Highest High in period) * 100
    wvf_period = 22
    highest_high = np.full(n, np.nan)
    for i in range(wvf_period - 1, n):
        highest_high[i] = np.max(high[i - wvf_period + 1:i + 1])
    
    wvf = np.full(n, np.nan)
    for i in range(wvf_period - 1, n):
        if highest_high[i] > 0:
            wvf[i] = ((highest_high[i] - low[i]) / highest_high[i]) * 100
        else:
            wvf[i] = 0
    
    # Volume confirmation
    vol_ma_period = 20
    vol_ma = np.full(n, np.nan)
    for i in range(vol_ma_period, n):
        vol_ma[i] = np.mean(volume[i-vol_ma_period:i])
    
    signals = np.zeros(n)
    position = 0
    size = 0.25  # 25% position size
    
    # Warmup: need WVF (22), EMA (50), volume MA (20)
    start_idx = max(wvf_period, ema_period, vol_ma_period)
    
    for i in range(start_idx, n):
        if (np.isnan(wvf[i]) or
            np.isnan(ema_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        # Trend filter: price above/below 1d EMA(50)
        uptrend = price > ema_aligned[i]
        downtrend = price < ema_aligned[i]
        
        # Volume confirmation: > 1.3x average volume
        volume_confirmation = vol_ratio > 1.3
        
        # WVF threshold for volatility/spike detection
        wvf_threshold = 0.8  # 80% of range
        
        if position == 0:
            # Long entry: WVF spike (fear) in uptrend with volume
            if wvf[i] > wvf_threshold and uptrend and volume_confirmation:
                signals[i] = size
                position = 1
            # Short entry: WVF spike (fear) in downtrend with volume
            elif wvf[i] > wvf_threshold and downtrend and volume_confirmation:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: WVF normalizes or trend reverses
            if wvf[i] < 0.3 or not uptrend:  # Exit when fear subsides or trend breaks
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: WVF normalizes or trend reverses
            if wvf[i] < 0.3 or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_WilliamsVixFix_MeanReversion_1dTrend"
timeframe = "6h"
leverage = 1.0