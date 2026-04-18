#!/usr/bin/env python3
"""
Hypothesis: 4h volume-weighted VWAP mean reversion with 1d RSI filter.
- Long: price < VWAP(20) and RSI(14) < 30 on 1d (oversold in larger trend)
- Short: price > VWAP(20) and RSI(14) > 70 on 1d (overbought in larger trend)
- Exit: price crosses back to VWAP(20) or RSI returns to neutral zone (40-60)
- Uses 4h VWAP for entry timing and 1d RSI for regime filter to avoid counter-trend trades.
Designed for 20-50 trades/year (80-200 total) to minimize fee drag while capturing mean reversion in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_vwap(high, low, close, volume, period):
    """Calculate Volume Weighted Average Price."""
    if len(close) < period:
        return np.full(len(close), np.nan)
    
    typical_price = (high + low + close) / 3
    vwap = np.full(len(close), np.nan)
    
    for i in range(period - 1, len(close)):
        tp_sum = np.sum(typical_price[i - period + 1:i + 1] * volume[i - period + 1:i + 1])
        vol_sum = np.sum(volume[i - period + 1:i + 1])
        if vol_sum != 0:
            vwap[i] = tp_sum / vol_sum
    
    return vwap

def calculate_rsi(close, period):
    """Calculate Relative Strength Index."""
    if len(close) < period + 1:
        return np.full(len(close), np.nan)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full(len(close), np.nan)
    avg_loss = np.full(len(close), np.nan)
    
    if len(gain) >= period:
        avg_gain[period] = np.mean(gain[:period])
        avg_loss[period] = np.mean(loss[:period])
        
        for i in range(period + 1, len(close)):
            avg_gain[i] = (avg_gain[i-1] * (period - 1) + gain[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period - 1) + loss[i-1]) / period
    
    rs = np.full(len(close), np.nan)
    for i in range(period, len(close)):
        if avg_loss[i] != 0:
            rs[i] = avg_gain[i] / avg_loss[i]
    
    rsi = np.full(len(close), np.nan)
    for i in range(period, len(close)):
        if rs[i] != 0:
            rsi[i] = 100 - (100 / (1 + rs[i]))
        else:
            rsi[i] = 100  # when avg_loss is 0, RSI is 100
    
    return rsi

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for RSI filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate VWAP (20-period) on 4h
    vwap_20 = calculate_vwap(high, low, close, volume, 20)
    
    # Calculate RSI (14-period) on 1d
    rsi_14_1d = calculate_rsi(close_1d, 14)
    
    # Align 1d RSI to 4h timeframe
    rsi_14_1d_4h = align_htf_to_ltf(prices, df_1d, rsi_14_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # need VWAP(20) and aligned RSI
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if np.isnan(vwap_20[i]) or np.isnan(rsi_14_1d_4h[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price below VWAP and 1d RSI oversold (<30)
            if close[i] < vwap_20[i] and rsi_14_1d_4h[i] < 30:
                signals[i] = 0.25
                position = 1
            # Short: price above VWAP and 1d RSI overbought (>70)
            elif close[i] > vwap_20[i] and rsi_14_1d_4h[i] > 70:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses back to VWAP or RSI returns to neutral (>=40)
            if close[i] >= vwap_20[i] or rsi_14_1d_4h[i] >= 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses back to VWAP or RSI returns to neutral (<=60)
            if close[i] <= vwap_20[i] or rsi_14_1d_4h[i] <= 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_VWAP20_RSI14_MeanReversion"
timeframe = "4h"
leverage = 1.0