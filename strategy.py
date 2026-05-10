#!/usr/bin/env python3
# 6h_VWAP_Rebound_Volume_Trend
# Hypothesis: Mean-reversion to VWAP with volume confirmation and 12h trend filter.
# Long when price pulls back to VWAP from below with volume surge and 12h uptrend.
# Short when price pulls back to VWAP from above with volume surge and 12h downtrend.
# Uses 12h EMA as trend filter to align with higher timeframe momentum.
# Designed for low trade frequency (target: 15-30 trades/year) to minimize fee drag.
# Works in bull markets via buying dips and in bear markets via selling rallies.

name = "6h_VWAP_Rebound_Volume_Trend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate 12h EMA30 for trend filter
    close_12h = df_12h['close'].values
    ema_30_12h = pd.Series(close_12h).ewm(span=30, adjust=False, min_periods=30).mean().values
    ema_30_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_30_12h)
    
    # 6h data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # VWAP calculation (typical price * volume)
    typical_price = (high + low + close) / 3.0
    vwap_numerator = pd.Series(typical_price * volume).rolling(window=24, min_periods=24).sum().values  # 24 periods = 6h * 4 = 1 day
    vwap_denominator = pd.Series(volume).rolling(window=24, min_periods=24).sum().values
    vwap = np.where(vwap_denominator != 0, vwap_numerator / vwap_denominator, typical_price)
    
    # Volume average (24-period)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need VWAP (24) + EMA (30)
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(vwap[i]) or np.isnan(vol_ma[i]) or
            np.isnan(ema_30_12h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine trend from 12h EMA
        close_12h_aligned = align_htf_to_ltf(prices, df_12h, close_12h)
        uptrend = close_12h_aligned[i] > ema_30_12h_aligned[i]
        downtrend = close_12h_aligned[i] < ema_30_12h_aligned[i]
        
        # Volume confirmation
        volume_surge = volume[i] > 1.8 * vol_ma[i]
        
        # Price position relative to VWAP
        price_above_vwap = close[i] > vwap[i]
        price_below_vwap = close[i] < vwap[i]
        
        if position == 0:
            # Long: pullback to VWAP from below + volume surge + 12h uptrend
            if price_below_vwap and close[i] >= vwap[i] * 0.998 and volume_surge and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: pullback to VWAP from above + volume surge + 12h downtrend
            elif price_above_vwap and close[i] <= vwap[i] * 1.002 and volume_surge and downtrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price moves above VWAP significantly OR trend changes
            if close[i] > vwap[i] * 1.015 or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price moves below VWAP significantly OR trend changes
            if close[i] < vwap[i] * 0.985 or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals