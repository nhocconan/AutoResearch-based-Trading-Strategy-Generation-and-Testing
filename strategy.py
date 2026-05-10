#!/usr/bin/env python3
# 6h_VWAP_Cross_With_1dTrend_and_VolumeSurge
# Hypothesis: In 6h timeframe, price often reverts to the session VWAP during pullbacks in a strong trend.
# We use 1-day EMA50 to determine the primary trend direction.
# We enter long when: (1) 1-day uptrend (close > EMA50_1d), (2) price crosses above 6h VWAP from below,
# (3) volume is above average (surge). Short rules are inverse.
# We exit when price crosses back below VWAP (for longs) or above VWAP (for shorts) or when trend changes.
# This strategy aims to capture trend continuation moves with high-probability entries during pullbacks,
# and should work in both bull and bear markets by following the higher timeframe trend.

name = "6h_VWAP_Cross_With_1dTrend_and_VolumeSurge"
timeframe = "6h"
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
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 6h VWAP (typical price * volume cumsum / volume cumsum)
    typical_price = (high + low + close) / 3.0
    tp_vol = typical_price * volume
    cum_tp_vol = np.nancumsum(tp_vol)
    cum_vol = np.nancumsum(volume)
    # Avoid division by zero
    vwap = np.divide(cum_tp_vol, cum_vol, out=np.full_like(cum_tp_vol, np.nan), where=cum_vol!=0)
    
    # Calculate daily EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume average (24-period = 4 days) for surge detection
    volume_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need enough data for VWAP (at least 1), EMA50_1d (50), volume MA (24)
    start_idx = max(1, 50, 24)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(vwap[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Daily trend filter
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Volume confirmation (surge)
        volume_surge = volume[i] > volume_ma[i] * 1.5
        
        # VWAP cross detection
        if i > 0:
            cross_above_vwap = (close[i] > vwap[i]) and (close[i-1] <= vwap[i-1])
            cross_below_vwap = (close[i] < vwap[i]) and (close[i-1] >= vwap[i-1])
        else:
            cross_above_vwap = False
            cross_below_vwap = False
        
        if position == 0:
            # Long entry: uptrend + VWAP cross above + volume surge
            if uptrend and cross_above_vwap and volume_surge:
                signals[i] = 0.25
                position = 1
            # Short entry: downtrend + VWAP cross below + volume surge
            elif downtrend and cross_below_vwap and volume_surge:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: trend breaks or VWAP cross below
            if not uptrend or cross_below_vwap:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: trend breaks or VWAP cross above
            if not downtrend or cross_above_vwap:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals