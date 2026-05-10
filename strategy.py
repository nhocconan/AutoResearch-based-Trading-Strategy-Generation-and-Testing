#!/usr/bin/env python3
# 6h_1d_1w_RollingReversal
# Hypothesis: 6h mean reversion when price deviates from 1d VWAP and aligns with weekly trend.
# Uses 1d VWAP as fair value and 1w EMA50 for trend filter. Enters when price reverts to VWAP
# with volume confirmation, exits on VWAP cross or trend change. Designed for low trade frequency
# (<30/year) to work in both bull and bear markets by fading extremes in trending markets.

name = "6h_1d_1w_RollingReversal"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data for VWAP and 1w data for trend
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1d) < 2 or len(df_1w) < 2:
        return np.zeros(n)
    
    # 1d VWAP calculation (typical price * volume)
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3.0
    vwap_numerator = (typical_price * df_1d['volume']).cumsum()
    vwap_denominator = df_1d['volume'].cumsum()
    vwap = (vwap_numerator / vwap_denominator).values
    
    # 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # 6h data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-calculate VWAP alignment and trend
    vwap_aligned = align_htf_to_ltf(prices, df_1d, vwap)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Price deviation from VWAP (normalized by ATR-like measure)
    price_dev = (close - vwap_aligned) / vwap_aligned
    
    # Volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need VWAP (1d), EMA (1w), volume MA (20)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(vwap_aligned[i]) or np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine trend from 1w EMA50
        uptrend = close[i] > ema_50_1w_aligned[i]
        downtrend = close[i] < ema_50_1w_aligned[i]
        
        # Volume confirmation
        volume_surge = volume[i] > 1.5 * vol_ma[i]
        
        # Mean reversion signals: price deviated from VWAP
        dev_long = price_dev[i] < -0.015  # 1.5% below VWAP
        dev_short = price_dev[i] > 0.015   # 1.5% above VWAP
        
        # Re-entry signals: price returning to VWAP
        reentry_long = (price_dev[i] > -0.005) and (price_dev[i-1] <= -0.005)
        reentry_short = (price_dev[i] < 0.005) and (price_dev[i-1] >= 0.005)
        
        if position == 0:
            # Long: price oversold below VWAP with volume surge and weekly uptrend
            if dev_long and volume_surge and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: price overbought above VWAP with volume surge and weekly downtrend
            elif dev_short and volume_surge and downtrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price returns to VWAP OR trend changes
            if reentry_long or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price returns to VWAP OR trend changes
            if reentry_short or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals