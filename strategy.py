#!/usr/bin/env python3
# 4h_1D_HTF_RSI_With_Triple_Filter
# Hypothesis: Use daily RSI (14) as primary directional filter on 4h chart. Enter long when daily RSI > 55 and 4h price closes above 4h EMA(20). Enter short when daily RSI < 45 and 4h price closes below 4h EMA(20). Add volume confirmation (1.5x 4-period volume MA) to reduce false signals. Exit on opposite RSI extreme or EMA crossover. Designed to work in both bull and bear markets by using higher timeframe momentum filter. Target: 20-40 trades/year to minimize fee drag.

timeframe = "4h"
name = "4h_1D_HTF_RSI_With_Triple_Filter"
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
    
    # Get daily data for RSI filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) == 0:
        return np.zeros(n)
    
    # Calculate daily RSI (14)
    delta = pd.Series(df_1d['close']).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d_vals = rsi_1d.values
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d_vals)
    
    # Calculate 4h EMA(20)
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Volume confirmation: 1.5x average volume (4-period = 1 day on 4h chart)
    vol_ma = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 4)  # Ensure we have EMA and volume MA data
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(ema_20[i]) or np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: daily RSI > 55, price closes above EMA(20), volume confirmation
            if (rsi_1d_aligned[i] > 55 and 
                close[i] > ema_20[i] and 
                volume[i] > 1.5 * vol_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short: daily RSI < 45, price closes below EMA(20), volume confirmation
            elif (rsi_1d_aligned[i] < 45 and 
                  close[i] < ema_20[i] and 
                  volume[i] > 1.5 * vol_ma[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: daily RSI < 45 OR price closes below EMA(20)
            if (rsi_1d_aligned[i] < 45 or close[i] < ema_20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: daily RSI > 55 OR price closes above EMA(20)
            if (rsi_1d_aligned[i] > 55 or close[i] > ema_20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals