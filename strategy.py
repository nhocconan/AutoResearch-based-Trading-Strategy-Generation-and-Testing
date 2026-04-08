#!/usr/bin/env python3
# 4h_camilla_pivot_volume_v2
# Hypothesis: Camarilla pivot levels (based on 1d high/low/close) provide mean reversion levels in ranging markets and breakout levels in trending markets.
# Long when: price crosses above Camarilla H3 level, 1d RSI > 50 (bullish bias), volume > 1.5x average.
# Short when: price crosses below Camarilla L3 level, 1d RSI < 50 (bearish bias), volume > 1.5x average.
# Exit when price returns to Camarilla H4/L4 levels or volume drops below average.
# Uses 1d timeframe for pivot levels and RSI bias, 4h for entry/execution.
# Target: 20-40 trades/year to avoid fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_camilla_pivot_volume_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume filter: 1.5x 20-period average
    vol_ma_period = 20
    vol_ma = np.full(n, np.nan)
    for i in range(vol_ma_period-1, n):
        vol_ma[i] = np.mean(volume[i-vol_ma_period+1:i+1])
    
    vol_surge = np.full(n, False)
    for i in range(n):
        if not np.isnan(vol_ma[i]) and vol_ma[i] > 0:
            vol_surge[i] = volume[i] > 1.5 * vol_ma[i]
    
    # Get 1d data for Camarilla pivots and RSI
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for each 1d bar
    # H4 = close + 1.5 * (high - low)
    # H3 = close + 1.1 * (high - low)
    # L3 = close - 1.1 * (high - low)
    # L4 = close - 1.5 * (high - low)
    camarilla_h4 = np.full(len(close_1d), np.nan)
    camarilla_h3 = np.full(len(close_1d), np.nan)
    camarilla_l3 = np.full(len(close_1d), np.nan)
    camarilla_l4 = np.full(len(close_1d), np.nan)
    
    for i in range(len(close_1d)):
        if not np.isnan(high_1d[i]) and not np.isnan(low_1d[i]) and not np.isnan(close_1d[i]):
            rng = high_1d[i] - low_1d[i]
            camarilla_h4[i] = close_1d[i] + 1.5 * rng
            camarilla_h3[i] = close_1d[i] + 1.1 * rng
            camarilla_l3[i] = close_1d[i] - 1.1 * rng
            camarilla_l4[i] = close_1d[i] - 1.5 * rng
    
    # Calculate 1d RSI for bias
    rsi_period = 14
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full(len(close_1d), np.nan)
    avg_loss = np.full(len(close_1d), np.nan)
    
    # Initial average
    if rsi_period < len(close_1d):
        avg_gain[rsi_period] = np.mean(gain[1:rsi_period+1])
        avg_loss[rsi_period] = np.mean(loss[1:rsi_period+1])
    
        # Wilder's smoothing
        for i in range(rsi_period+1, len(close_1d)):
            avg_gain[i] = (avg_gain[i-1] * (rsi_period-1) + gain[i]) / rsi_period
            avg_loss[i] = (avg_loss[i-1] * (rsi_period-1) + loss[i]) / rsi_period
    
    rs = np.full(len(close_1d), np.nan)
    rsi_1d = np.full(len(close_1d), np.nan)
    for i in range(rsi_period, len(close_1d)):
        if avg_loss[i] != 0:
            rs[i] = avg_gain[i] / avg_loss[i]
            rsi_1d[i] = 100 - (100 / (1 + rs[i]))
        else:
            rsi_1d[i] = 100 if avg_gain[i] != 0 else 50
    
    # Align 1d data to 4h
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = max(vol_ma_period, 1)  # Need at least volume MA and 1 bar for crossover
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(vol_ma[i]) or np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or np.isnan(rsi_1d_aligned[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Price returns to H4 level or volume drops below average
            if close[i] <= camarilla_h4_aligned[i] or volume[i] < vol_ma[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price returns to L4 level or volume drops below average
            if close[i] >= camarilla_l4_aligned[i] or volume[i] < vol_ma[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: Price crosses above H3, 1d RSI > 50, volume surge
            if (close[i] > camarilla_h3_aligned[i] and close[i-1] <= camarilla_h3_aligned[i-1] and
                rsi_1d_aligned[i] > 50 and vol_surge[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: Price crosses below L3, 1d RSI < 50, volume surge
            elif (close[i] < camarilla_l3_aligned[i] and close[i-1] >= camarilla_l3_aligned[i-1] and
                  rsi_1d_aligned[i] < 50 and vol_surge[i]):
                position = -1
                signals[i] = -0.25
    
    return signals