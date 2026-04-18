#!/usr/bin/env python3
"""
Hypothesis: 1d Volume-weighted average price (VWAP) with 1w trend filter.
In bull markets: price above 1w VWAP acts as support, buy on dips with volume.
In bear markets: price below 1w VWAP acts as resistance, sell on rallies with volume.
Weekly VWAP avoids whipsaw: only trade in direction of weekly trend.
Designed for 10-20 trades/year to minimize fee drift.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_vwap(high, low, close, volume):
    """Calculate Volume Weighted Average Price."""
    typical_price = (high + low + close) / 3.0
    vwap = np.full(len(close), np.nan)
    cum_vol = np.cumsum(volume)
    cum_tpv = np.cumsum(typical_price * volume)
    # Avoid division by zero
    mask = cum_vol != 0
    vwap[mask] = cum_tpv[mask] / cum_vol[mask]
    return vwap

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for VWAP
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate VWAP on 1d
    vwap_1d = calculate_vwap(high_1d, low_1d, close_1d, volume_1d)
    
    # Calculate EMA(20) on 1w for trend filter
    ema_20_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 20:
        ema_20_1w[19] = np.mean(close_1w[:20])
        for i in range(20, len(close_1w)):
            ema_20_1w[i] = (close_1w[i] * 2 / (20 + 1)) + ema_20_1w[i-1] * (1 - 2 / (20 + 1))
    
    # Align to 1d timeframe (same as primary)
    vwap_1d_aligned = vwap_1d  # Already on 1d
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # need EMA calculation
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(vwap_1d_aligned[i]) or np.isnan(ema_20_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above VWAP, above weekly EMA (bullish trend)
            if close[i] > vwap_1d_aligned[i] and close[i] > ema_20_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price below VWAP, below weekly EMA (bearish trend)
            elif close[i] < vwap_1d_aligned[i] and close[i] < ema_20_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below VWAP or below weekly EMA
            if close[i] <= vwap_1d_aligned[i] or close[i] <= ema_20_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above VWAP or above weekly EMA
            if close[i] >= vwap_1d_aligned[i] or close[i] >= ema_20_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_VWAP_1wEMA20_Trend"
timeframe = "1d"
leverage = 1.0