#!/usr/bin/env python3
# 12h_Volume_Weighted_VWAP_Reversion
# Hypothesis: Price reverts to volume-weighted average price (VWAP) over the past 12 hours. 
# Long when price deviates below VWAP by >1.5 standard deviations with volume confirmation.
# Short when price deviates above VWAP by >1.5 standard deviations with volume confirmation.
# Uses 1-day trend filter (EMA50) to align with higher timeframe momentum and avoid counter-trend trades.
# Designed for low frequency (~20-40 trades/year) to minimize fee drag on 12h timeframe.

name = "12h_Volume_Weighted_VWAP_Reversion"
timeframe = "12h"
leverage = 1.0

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
    
    # Get 1-day data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 12-period VWAP and standard deviation (using typical price)
    typical_price = (high + low + close) / 3.0
    pv = typical_price * volume
    cum_pv = np.nancumsum(pv)
    cum_vol = np.nancumsum(volume)
    vwap = np.where(cum_vol != 0, cum_pv / cum_vol, typical_price)
    
    # Calculate rolling standard deviation of price deviation from VWAP
    price_dev = typical_price - vwap
    # Use pandas rolling for std with min_periods
    price_dev_series = pd.Series(price_dev)
    vwap_std = price_dev_series.rolling(window=12, min_periods=12).std().values
    
    # Volume confirmation: current volume > 1.5x 12-period volume MA
    vol_ma_12 = pd.Series(volume).rolling(window=12, min_periods=12).mean().values
    volume_confirm = volume > vol_ma_12 * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need EMA50 (50), VWAP std (12), volume MA (12)
    start_idx = max(50, 12)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vwap_std[i]) or 
            np.isnan(vwap[i]) or 
            np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # 1-day trend filter
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Calculate z-score of price deviation from VWAP
        if vwap_std[i] > 0:
            z_score = price_dev[i] / vwap_std[i]
        else:
            z_score = 0
        
        if position == 0:
            # Long entry: price below VWAP (-z-score) with volume confirmation in uptrend
            if z_score < -1.5 and volume_confirm[i] and uptrend:
                signals[i] = 0.25
                position = 1
            # Short entry: price above VWAP (+z-score) with volume confirmation in downtrend
            elif z_score > 1.5 and volume_confirm[i] and downtrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price reverts to VWAP (z-score > -0.5) or trend breaks
            if z_score > -0.5 or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price reverts to VWAP (z-score < 0.5) or trend breaks
            if z_score < 0.5 or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals