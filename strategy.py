#!/usr/bin/env python3
"""
4H_VWAP_MeanReversion_1dTrend_Volume
Hypothesis: Uses VWAP deviation from 4h VWAP (price crossing above/below VWAP) for mean reversion entries,
confirmed by 1d EMA trend and volume spike. Designed for 4h timeframe to capture mean reversion moves
with low trade frequency (target: 20-40 trades/year). Works in both bull and bear markets by following
1d trend direction (long only in uptrend, short only in downtrend), avoiding counter-trend trades.
Uses discrete position sizing (0.25) to minimize fee churn.
"""

name = "4H_VWAP_MeanReversion_1dTrend_Volume"
timeframe = "4h"
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
    
    # Get 1d data for EMA trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA(50) for trend direction
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 4h VWAP
    typical_price = (high + low + close) / 3.0
    vwap_numerator = np.cumsum(typical_price * volume)
    vwap_denominator = np.cumsum(volume)
    vwap = np.where(vwap_denominator != 0, vwap_numerator / vwap_denominator, np.nan)
    
    # Volume filter: volume > 2.0x 20-period average on 4h chart
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_threshold = vol_ma * 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Warmup for EMA
    
    for i in range(start_idx, n):
        if np.isnan(ema_1d_aligned[i]) or np.isnan(vwap[i]) or np.isnan(vol_threshold[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: price above/below 1d EMA50
        price_above_ema = close[i] > ema_1d_aligned[i]
        price_below_ema = close[i] < ema_1d_aligned[i]
        
        if position == 0:
            # Long entry: price crosses above VWAP + above 1d EMA + volume spike
            if (close[i] > vwap[i] and 
                close[i-1] <= vwap[i-1] and  # crossed above VWAP
                price_above_ema and 
                volume[i] > vol_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price crosses below VWAP + below 1d EMA + volume spike
            elif (close[i] < vwap[i] and 
                  close[i-1] >= vwap[i-1] and  # crossed below VWAP
                  price_below_ema and 
                  volume[i] > vol_threshold[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below VWAP or volume drops
            if (close[i] < vwap[i] and close[i-1] >= vwap[i-1]) or volume[i] < vol_ma[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above VWAP or volume drops
            if (close[i] > vwap[i] and close[i-1] <= vwap[i-1]) or volume[i] < vol_ma[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals