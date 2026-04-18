#!/usr/bin/env python3
"""
6h_OrderBlock_OrderFlow_Imbalance
6h strategy using order block detection and order flow imbalance from volume delta.
- Long: Bullish order block + positive volume delta + price above 1w VWAP
- Short: Bearish order block + negative volume delta + price below 1w VWAP
- Exit: Opposite signal or price crosses 1w VWAP in opposite direction
Designed for ~20-30 trades/year per symbol (80-120 total over 4 years)
Works in bull markets (accumulation/demand zones) and bear markets (distribution/supply zones)
"""

import numpy as np
import pandas as pd
from mtrf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for VWAP and structure
    df_1w = get_htf_data(prices, '1w')
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Weekly VWAP (Volume Weighted Average Price)
    typical_price_1w = (high_1w + low_1w + close_1w) / 3.0
    vwap_1w = np.cumsum(typical_price_1w * volume_1w) / np.cumsum(volume_1w)
    vwap_1w = np.where(np.cumsum(volume_1w) == 0, 0, vwap_1w)
    
    vwap_1w_aligned = align_htf_to_ltf(prices, df_1w, vwap_1w)
    
    # Daily data for volume delta calculation
    df_1d = get_htf_data(prices, '1d')
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Approximate volume delta using close position in daily range
    # In real implementation, this would use tick data, but we approximate:
    # If close > midpoint of range, buying pressure; if close < midpoint, selling pressure
    daily_midpoint = (high_1d + low_1d) / 2.0
    volume_delta_approx = np.where(close_1d > daily_midpoint, volume_1d, -volume_1d)
    
    # Smooth the volume delta to get order flow imbalance
    vol_delta_smooth = pd.Series(volume_delta_approx).ewm(span=21, adjust=False, min_periods=21).mean().values
    vol_delta_aligned = align_htf_to_ltf(prices, df_1d, vol_delta_smooth)
    
    # Detect order blocks (simplified: strong candle opposite to recent trend)
    # Bullish order block: strong down candle followed by up move
    # Bearish order block: strong up candle followed by down move
    body_size = np.abs(close - open_) if 'open_' in locals() else np.abs(close - high)  # fallback
    if 'open_' not in locals():
        open_ = prices['open'].values
    body_size = np.abs(close - open_)
    candle_range = high - low
    strong_candle = candle_range > 0  # avoid division by zero
    body_ratio = np.where(candle_range > 0, body_size / candle_range, 0)
    
    # Bullish OB: bearish candle (close < open) with strong body, followed by bullish candle
    bearish_candle = close < open_
    bullish_candle = close > open_
    
    # Shift to get previous candle
    bearish_prev = np.roll(bearish_candle, 1)
    bullish_prev = np.roll(bullish_candle, 1)
    bearish_prev[0] = False
    bullish_prev[0] = False
    
    bullish_ob = bearish_prev & bullish_candle & (body_ratio > 0.6)
    bearish_ob = bullish_prev & bearish_candle & (body_ratio > 0.6)
    
    bullish_ob_aligned = align_htf_to_ltf(prices, df_1d, bullish_ob.astype(float))
    bearish_ob_aligned = align_htf_to_ltf(prices, df_1d, bearish_ob.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # need enough for EMA smoothing
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(vwap_1w_aligned[i]) or np.isnan(vol_delta_aligned[i]) or
            np.isnan(bullish_ob_aligned[i]) or np.isnan(bearish_ob_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Order flow conditions
        positive_flow = vol_delta_aligned[i] > 0
        negative_flow = vol_delta_aligned[i] < 0
        
        # Order block signals
        ob_bullish = bullish_ob_aligned[i] > 0.5
        ob_bearish = bearish_ob_aligned[i] > 0.5
        
        # VWAP position
        above_vwap = close[i] > vwap_1w_aligned[i]
        below_vwap = close[i] < vwap_1w_aligned[i]
        
        if position == 0:
            # Long: bullish OB + positive flow + above VWAP
            if ob_bullish and positive_flow and above_vwap:
                signals[i] = 0.25
                position = 1
            # Short: bearish OB + negative flow + below VWAP
            elif ob_bearish and negative_flow and below_vwap:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: bearish OB with negative flow OR price crosses below VWAP
            if (ob_bearish and negative_flow) or below_vwap:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: bullish OB with positive flow OR price crosses above VWAP
            if (ob_bullish and positive_flow) or above_vwap:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_OrderBlock_OrderFlow_Imbalance"
timeframe = "6h"
leverage = 1.0