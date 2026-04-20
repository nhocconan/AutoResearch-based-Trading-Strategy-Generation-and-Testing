#!/usr/bin/env python3
"""
4h_1D_RelativeStrength_Index_Divergence
Hypothesis: Trade 4h momentum reversals identified by RSI divergence with 1d price action.
Long when 4h RSI shows bullish divergence (higher low in RSI, lower low in price) during 1d uptrend.
Short when 4h RSI shows bearish divergence (lower high in RSI, higher high in price) during 1d downtrend.
Uses RSI(14) on 4h with divergence detection and 1d EMA50 trend filter.
Designed for 4h timeframe to capture medium-term reversals with controlled frequency.
Target: 15-35 trades/year (60-140 total) with position size 0.25.
Works in bull/bear: 1d trend filter ensures trades align with higher timeframe momentum.
"""

name = "4h_1D_RelativeStrength_Index_Divergence"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def rsi(close, period=14):
    """Calculate Relative Strength Index."""
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(close)
    avg_loss = np.zeros_like(close)
    
    # Wilder's smoothing
    avg_gain[period] = np.mean(gain[:period])
    avg_loss[period] = np.mean(loss[:period])
    
    for i in range(period + 1, len(close)):
        avg_gain[i] = (avg_gain[i-1] * (period - 1) + gain[i-1]) / period
        avg_loss[i] = (avg_loss[i-1] * (period - 1) + loss[i-1]) / period
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_values = np.zeros_like(close)
    rsi_values[:] = 50  # neutral
    rsi_values[period:] = 100 - (100 / (1 + rs[period:]))
    return rsi_values

def find_divergences(price, rsi_vals, lookback=10):
    """
    Find bullish and bearish divergences.
    Returns arrays with 1 for bullish div, -1 for bearish div, 0 otherwise.
    """
    n = len(price)
    bullish_div = np.zeros(n)
    bearish_div = np.zeros(n)
    
    for i in range(lookback, n):
        # Look for bullish divergence: price makes lower low, RSI makes higher low
        if i >= lookback * 2:  # need enough history
            # Find recent lows in price and RSI
            price_slice = price[i-lookback:i+1]
            rsi_slice = rsi_vals[i-lookback:i+1]
            
            # Simple peak/trough detection
            price_min_idx = np.argmin(price_slice)
            rsi_min_idx = np.argmin(rsi_slice)
            
            # Bullish divergence: price lower low, RSI higher low
            if (price_min_idx == lookback and  # recent price low
                rsi_min_idx < lookback and     # earlier RSI low
                price[i] < price[i-lookback] and
                rsi_vals[i] > rsi_vals[i-lookback]):
                bullish_div[i] = 1
                
            # Bearish divergence: price higher high, RSI lower high
            price_max_idx = np.argmax(price_slice)
            rsi_max_idx = np.argmax(rsi_slice)
            if (price_max_idx == lookback and  # recent price high
                rsi_max_idx < lookback and     # earlier RSI high
                price[i] > price[i-lookback] and
                rsi_vals[i] < rsi_vals[i-lookback]):
                bearish_div[i] = -1
    
    return bullish_div, bearish_div

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data for RSI calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    
    # Calculate 4h RSI(14)
    rsi_4h = rsi(close_4h, 14)
    rsi_4h_aligned = align_htf_to_ltf(prices, df_4h, rsi_4h)
    
    # Find divergences on 4h RSI
    bullish_div, bearish_div = find_divergences(close_4h, rsi_4h, lookback=10)
    bullish_div_aligned = align_htf_to_ltf(prices, df_4h, bullish_div)
    bearish_div_aligned = align_htf_to_ltf(prices, df_4h, bearish_div)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 for trend filter
    def ema(values, period):
        result = np.full_like(values, np.nan)
        if len(values) >= period:
            multiplier = 2.0 / (period + 1)
            result[period-1] = np.mean(values[:period])
            for i in range(period, len(values)):
                result[i] = multiplier * values[i] + (1 - multiplier) * result[i-1]
        return result
    
    ema50_1d = ema(close_1d, 50)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(rsi_4h_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or
            np.isnan(close[i]) or np.isnan(bullish_div_aligned[i]) or
            np.isnan(bearish_div_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: bullish RSI divergence during 1d uptrend (close > EMA50)
            if bullish_div_aligned[i] == 1 and close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: bearish RSI divergence during 1d downtrend (close < EMA50)
            elif bearish_div_aligned[i] == -1 and close[i] < ema50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: bearish divergence OR price breaks below EMA50
            if bearish_div_aligned[i] == -1 or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: bullish divergence OR price breaks above EMA50
            if bullish_div_aligned[i] == 1 or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals