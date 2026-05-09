#!/usr/bin/env python3
# Hypothesis: 6h timeframe using weekly VWAP as dynamic support/resistance with daily RSI filter for trend.
# Weekly VWAP acts as institutional reference point - price tends to revert to or break through with conviction.
# Daily RSI(14) filter ensures trades align with higher timeframe momentum, reducing whipsaw in ranging markets.
# Target: 80-150 total trades over 4 years (20-38/year) with size 0.25.

name = "6h_WeeklyVWAP_RSI14_Filter"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate weekly VWAP from prior week (42 bars = 7 days * 6 bars per day)
    # VWAP = sum(price * volume) / sum(volume)
    typical_price = (high + low + close) / 3.0
    pv = typical_price * volume
    
    # Roll back 42 bars to get prior week's data
    pv_sum = np.zeros(n)
    vol_sum = np.zeros(n)
    
    for i in range(42, n):
        pv_sum[i] = np.sum(pv[i-42:i])
        vol_sum[i] = np.sum(volume[i-42:i])
    
    weekly_vwap = np.full(n, np.nan)
    valid = vol_sum != 0
    weekly_vwap[valid] = pv_sum[valid] / vol_sum[valid]
    
    # Get daily data for RSI(14) filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate daily RSI(14)
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[13] = np.mean(gain[1:14])  # First average
    avg_loss[13] = np.mean(loss[1:14])
    
    for i in range(14, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi_1d = np.where(avg_loss == 0, 100, 100 - (100 / (1 + rs)))
    
    # Align RSI to 6h timeframe
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Define conditions
    price_above_vwap = close > weekly_vwap
    price_below_vwap = close < weekly_vwap
    rsi_bullish = rsi_1d_aligned > 50
    rsi_bearish = rsi_1d_aligned < 50
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 42)  # Ensure sufficient data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(weekly_vwap[i]) or np.isnan(rsi_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above weekly VWAP + daily RSI bullish (>50)
            if price_above_vwap[i] and rsi_bullish[i]:
                signals[i] = 0.25
                position = 1
            # Short: price below weekly VWAP + daily RSI bearish (<50)
            elif price_below_vwap[i] and rsi_bearish[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below weekly VWAP or RSI turns bearish
            if price_below_vwap[i] or not rsi_bullish[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above weekly VWAP or RSI turns bullish
            if price_above_vwap[i] or not rsi_bearish[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals