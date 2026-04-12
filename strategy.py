#!/usr/bin/env python3
"""
6h_1d_volume_weighted_rsi_mean_reversion
Hypothesis: Mean reversion on 6-hour timeframe using volume-weighted RSI(14) from 1-day timeframe.
In ranging markets (2025+), price tends to revert to VWAP. Volume-weighted RSI identifies
overextended conditions with institutional participation. Only trade when volume confirms.
Uses 1-day VWAP as dynamic mean and volume-weighted RSI extremes for entry.
Designed for low frequency (15-25 trades/year) to minimize fee drift in choppy markets.
Works in both bull/bear by fading extremes regardless of trend direction.
"""

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
    
    # Get 1-day data for VWAP and volume-weighted RSI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1-day VWAP (volume-weighted average price)
    typical_price_1d = (high_1d + low_1d + close_1d) / 3.0
    vwap_numerator = np.cumsum(typical_price_1d * volume_1d)
    vwap_denominator = np.cumsum(volume_1d)
    vwap_1d = np.where(vwap_denominator > 0, vwap_numerator / vwap_denominator, typical_price_1d)
    
    # Calculate volume-weighted RSI(14) on 1-day timeframe
    # Weight price changes by volume
    delta = np.diff(close_1d, prepend=close_1d[0])
    weighted_gain = np.where(delta > 0, delta * volume_1d, 0.0)
    weighted_loss = np.where(delta < 0, -delta * volume_1d, 0.0)
    
    # Smoothed weighted gains/losses using Wilder's smoothing
    avg_weighted_gain = np.zeros_like(weighted_gain)
    avg_weighted_loss = np.zeros_like(weighted_loss)
    
    for i in range(1, len(weighted_gain)):
        if i < 14:
            # Simple average for first 14 periods
            avg_weighted_gain[i] = np.mean(weighted_gain[max(0, i-13):i+1])
            avg_weighted_loss[i] = np.mean(weighted_loss[max(0, i-13):i+1])
        else:
            # Wilder's smoothing
            avg_weighted_gain[i] = (avg_weighted_gain[i-1] * 13 + weighted_gain[i]) / 14
            avg_weighted_loss[i] = (avg_weighted_loss[i-1] * 13 + weighted_loss[i]) / 14
    
    # Avoid division by zero
    rs = np.where(avg_weighted_loss != 0, avg_weighted_gain / avg_weighted_loss, 0)
    vw_rsi_1d = 100 - (100 / (1 + rs))
    
    # Align VWAP and volume-weighted RSI to 6h timeframe
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    vw_rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, vw_rsi_1d)
    
    signals = np.zeros(n)
    
    for i in range(30, n):
        # Skip if data not ready
        if (np.isnan(vwap_1d_aligned[i]) or np.isnan(vw_rsi_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Mean reversion signals based on volume-weighted RSI extremes
        # Oversold: VW-RSI < 25 -> long
        # Overbought: VW-RSI > 75 -> short
        if vw_rsi_1d_aligned[i] < 25 and close[i] < vwap_1d_aligned[i]:
            # Long: price below VWAP and oversold
            signals[i] = 0.25
        elif vw_rsi_1d_aligned[i] > 75 and close[i] > vwap_1d_aligned[i]:
            # Short: price above VWAP and overbought
            signals[i] = -0.25
        else:
            # No clear signal - stay flat or hold
            signals[i] = 0.0
    
    return signals

name = "6h_1d_volume_weighted_rsi_mean_reversion"
timeframe = "6h"
leverage = 1.0