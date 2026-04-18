#!/usr/bin/env python3
"""
4h_Daily_RSI_MeanReversion_With_Volume_Spike
Hypothesis: In mean-reverting markets (chop regime), daily RSI extremes combined with volume spikes signal reversals. Uses 1D RSI(14) for overbought/oversold detection and 4H volume > 2x 20-period average for confirmation. Works in both bull and bear markets by fading extremes during consolidation periods. Targets 20-30 trades/year with position size 0.25.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1D data for RSI calculation
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1D RSI(14)
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full_like(close_1d, np.nan)
    avg_loss = np.full_like(close_1d, np.nan)
    
    # Wilder's smoothing
    for i in range(len(close_1d)):
        if i < 14:
            if i == 0:
                avg_gain[i] = gain[i]
                avg_loss[i] = loss[i]
            else:
                avg_gain[i] = (avg_gain[i-1] * i + gain[i]) / (i + 1)
                avg_loss[i] = (avg_loss[i-1] * i + loss[i]) / (i + 1)
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi_1d = 100 - (100 / (1 + rs))
    
    # Align RSI to 4h timeframe (wait for daily bar close)
    rsi_4h = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Calculate volume average (20-period) for confirmation
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # need volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(rsi_4h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 2.0 * 20-period average
        vol_spike = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long entry: RSI < 30 (oversold) with volume spike
            if rsi_4h[i] < 30 and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short entry: RSI > 70 (overbought) with volume spike
            elif rsi_4h[i] > 70 and vol_spike:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: RSI crosses above 50 (mean reversion complete)
            if rsi_4h[i] > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: RSI crosses below 50 (mean reversion complete)
            if rsi_4h[i] < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Daily_RSI_MeanReversion_With_Volume_Spike"
timeframe = "4h"
leverage = 1.0