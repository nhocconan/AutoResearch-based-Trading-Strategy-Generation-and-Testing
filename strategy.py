#!/usr/bin/env python3
# 4h_12h_Volume_Weighted_Price_Action
# Hypothesis: On 4h timeframe, use 12h volume-weighted average price (VWAP) as dynamic support/resistance.
# Go long when price crosses above 12h VWAP with volume > 1.5x 20-period average.
# Go short when price crosses below 12h VWAP with volume > 1.5x 20-period average.
# Exit when price crosses back over 12h VWAP.
# Uses volume confirmation to avoid false breakouts and works in both trending and ranging markets.
# Target: 20-40 trades/year to minimize fee drag.

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
    
    # Load 12h data for VWAP calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate 12h VWAP (typical price * volume / cumulative volume)
    typical_price_12h = (high_12h + low_12h + close_12h) / 3
    vwap_12h = np.full_like(close_12h, np.nan)
    cum_vol = np.cumsum(volume_12h)
    cum_tpv = np.cumsum(typical_price_12h * volume_12h)
    vwap_12h = np.where(cum_vol > 0, cum_tpv / cum_vol, np.nan)
    
    # Align 12h VWAP to 4h timeframe
    vwap_12h_aligned = align_htf_to_ltf(prices, df_12h, vwap_12h)
    
    # Calculate 20-period volume moving average on 4h
    vol_ma_20 = np.full_like(volume, np.nan)
    for i in range(19, len(volume)):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):
        # Skip if VWAP is not available
        if np.isnan(vwap_12h_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Calculate volume ratio
        if np.isnan(vol_ma_20[i]) or vol_ma_20[i] <= 0:
            volume_ratio = 0
        else:
            volume_ratio = volume[i] / vol_ma_20[i]
        
        if position == 0:
            # Look for long entry: price crosses above VWAP with volume surge
            if (close[i] > vwap_12h_aligned[i] and 
                volume_ratio > 1.5):
                position = 1
                signals[i] = position_size
            # Look for short entry: price crosses below VWAP with volume surge
            elif (close[i] < vwap_12h_aligned[i] and 
                  volume_ratio > 1.5):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses back below VWAP
            if close[i] < vwap_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses back above VWAP
            if close[i] > vwap_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_12h_Volume_Weighted_Price_Action"
timeframe = "4h"
leverage = 1.0