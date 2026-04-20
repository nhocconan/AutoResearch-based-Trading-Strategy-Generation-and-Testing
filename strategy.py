#!/usr/bin/env python3
# 6h_12h_1d_VWAP_Trend_Conditioned_Breakout_V1
# Hypothesis: On 6h timeframe, trade breakouts from 12h VWAP bands with 1d trend filter.
# In bull/bear markets, price tends to revert to VWAP before breaking out. Use 1d EMA50 to filter trend direction.
# Only take long when price > 1d EMA50, short when price < 1d EMA50. Requires volume confirmation.
# Targets 20-40 trades/year by requiring VWAP deviation, volume spike, and trend alignment.
# Works in trending and ranging markets due to dynamic VWAP bands and trend filter.

name = "6h_12h_1d_VWAP_Trend_Conditioned_Breakout_V1"
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
    
    # Get 12h data ONCE before loop for VWAP bands
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Get 1d data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 12h VWAP and standard deviation bands
    typical_price_12h = (df_12h['high'] + df_12h['low'] + df_12h['close']) / 3
    vwap_num = (typical_price_12h * df_12h['volume']).cumsum()
    vwap_den = df_12h['volume'].cumsum()
    vwap = vwap_num / vwap_den
    
    # VWAP deviation and bands (2 standard deviations)
    vwap_dev = typical_price_12h - vwap
    vwap_var = (vwap_dev ** 2 * df_12h['volume']).cumsum() / vwap_den
    vwap_std = np.sqrt(vwap_var)
    upper_band = vwap + 2.0 * vwap_std
    lower_band = vwap - 2.0 * vwap_std
    
    # Align VWAP bands to 6h timeframe
    vwap_aligned = align_htf_to_ltf(prices, df_12h, vwap.values)
    upper_aligned = align_htf_to_ltf(prices, df_12h, upper_band.values)
    lower_aligned = align_htf_to_ltf(prices, df_12h, lower_band.values)
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume average for spike detection (20-period)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(vwap_aligned[i]) or np.isnan(upper_aligned[i]) or 
            np.isnan(lower_aligned[i]) or np.isnan(ema_50_aligned[i]) or
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above upper VWAP band in uptrend (price > EMA50)
            if (close[i] > upper_aligned[i] and 
                close[i] > ema_50_aligned[i] and
                volume[i] > 2.0 * volume_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower VWAP band in downtrend (price < EMA50)
            elif (close[i] < lower_aligned[i] and 
                  close[i] < ema_50_aligned[i] and
                  volume[i] > 2.0 * volume_ma[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns to VWAP or trend changes
            if close[i] <= vwap_aligned[i] or close[i] < ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to VWAP or trend changes
            if close[i] >= vwap_aligned[i] or close[i] > ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals