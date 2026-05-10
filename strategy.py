#!/usr/bin/env python3
"""
1h_VWAP_Breakout_4hTrend_1dVolFilter
Hypothesis: Combining 1h VWAP breakouts with 4h trend filter and 1d volume confirmation captures institutional momentum moves while filtering noise. VWAP acts as dynamic support/resistance, and requiring alignment across timeframes reduces false signals. Works in both bull/bear by following the higher timeframe trend.
"""

name = "1h_VWAP_Breakout_4hTrend_1dVolFilter"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # Get 1d data for volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 4h EMA34 for trend
    close_4h = df_4h['close'].values
    ema34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema34_4h)
    
    # Calculate 1d volume SMA20 for volume filter
    volume_1d = df_1d['volume'].values
    vol_sma20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_sma20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_sma20_1d)
    
    # 1h data for VWAP and signal generation
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate VWAP (typical price * volume cumulative)
    typical_price = (high + low + close) / 3.0
    vwap_numerator = np.cumsum(typical_price * volume)
    vwap_denominator = np.cumsum(volume)
    vwap = vwap_numerator / vwap_denominator
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need 4h EMA34 (34 periods) and 1d volume SMA (20 periods)
    start_idx = 34  # 4h EMA needs 34 periods
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema34_4h_aligned[i]) or 
            np.isnan(vol_sma20_1d_aligned[i]) or
            np.isnan(vwap[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: price vs 4h EMA34
        uptrend_4h = close[i] > ema34_4h_aligned[i]
        downtrend_4h = close[i] < ema34_4h_aligned[i]
        
        # Volume filter: current 1h volume > 1.5x 1d average volume (scaled)
        # Approximate 1d volume per 1h bar
        vol_filter = volume[i] > (vol_sma20_1d_aligned[i] / 24.0) * 1.5
        
        if position == 0:
            # Long: price crosses above VWAP with uptrend and volume
            if close[i] > vwap[i] and uptrend_4h and vol_filter:
                signals[i] = 0.20
                position = 1
            # Short: price crosses below VWAP with downtrend and volume
            elif close[i] < vwap[i] and downtrend_4h and vol_filter:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: price crosses below VWAP or trend fails
            if close[i] < vwap[i] or not uptrend_4h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: price crosses above VWAP or trend fails
            if close[i] > vwap[i] or not downtrend_4h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals