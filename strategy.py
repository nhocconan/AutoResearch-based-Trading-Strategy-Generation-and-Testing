#!/usr/bin/env python3
# 6h_hull_rsi_divergence_1d_trend_volume_v1
# Hypothesis: On 6h timeframe, use Hull Moving Average (HMA) crossovers with RSI divergence on 1d timeframe for signal confirmation.
# Long when: 6H HMA(9) crosses above HMA(21), 1D RSI shows bullish divergence (price makes lower low but RSI makes higher low), and volume > 1.5x average.
# Short when: 6H HMA(9) crosses below HMA(21), 1D RSI shows bearish divergence (price makes higher high but RSI makes lower high), and volume > 1.5x average.
# Exit when opposite HMA crossover occurs or volume drops below average.
# RSI divergence is calculated on daily timeframe to avoid noise and provide higher probability signals.
# Works in both bull and bear markets: HMA catches trends, RSI divergence filters for exhaustion points, volume confirms conviction.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_hull_rsi_divergence_1d_trend_volume_v1"
timeframe = "6h"
leverage = 1.0

def hull_moving_average(arr, period):
    """Calculate Hull Moving Average"""
    if len(arr) < period:
        return np.full_like(arr, np.nan)
    half = int(period / 2)
    sqrt = int(np.sqrt(period))
    wma_half = pd.Series(arr).ewm(span=half, adjust=False).mean()
    wma_full = pd.Series(arr).ewm(span=period, adjust=False).mean()
    raw_hma = 2 * wma_half - wma_full
    hma = pd.Series(raw_hma).ewm(span=sqrt, adjust=False).mean()
    return hma.values

def calculate_rsi(prices, period=14):
    """Calculate RSI"""
    if len(prices) < period + 1:
        return np.full_like(prices, np.nan)
    delta = np.diff(prices)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return np.concatenate([[np.nan], rsi.values])

def find_divergence(price, indicator, lookback=5):
    """
    Find bullish/bearish divergence
    Returns: 1 for bullish div, -1 for bearish div, 0 for none
    """
    if len(price) < lookback or len(indicator) < lookback:
        return 0
    
    # Check for bullish divergence: price makes lower low, indicator makes higher low
    price_low_idx = np.argmin(price[-lookback:])
    indicator_low_idx = np.argmin(indicator[-lookback:])
    
    bullish_div = 0
    if price_low_idx == lookback - 1 and indicator_low_idx != lookback - 1:
        # Price made new low recently, indicator did not
        if indicator[-1] > indicator[indicator_low_idx]:
            bullish_div = 1
    
    # Check for bearish divergence: price makes higher high, indicator makes lower high
    price_high_idx = np.argmax(price[-lookback:])
    indicator_high_idx = np.argmax(indicator[-lookback:])
    
    bearish_div = 0
    if price_high_idx == lookback - 1 and indicator_high_idx != lookback - 1:
        # Price made new high recently, indicator did not
        if indicator[-1] < indicator[indicator_high_idx]:
            bearish_div = -1
            
    return bullish_div + bearish_div  # Will be 1, -1, or 0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 6H HMA for crossover signals
    hma_fast = hull_moving_average(close, 9)
    hma_slow = hull_moving_average(close, 21)
    
    # 1D data for RSI divergence
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1D RSI
    daily_close = df_1d['close'].values
    daily_rsi = calculate_rsi(daily_close, 14)
    
    # Align 1D RSI to 6H timeframe
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, daily_rsi)
    
    # Volume confirmation: 20-period average on 6H
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if data not available
        if np.isnan(hma_fast[i]) or np.isnan(hma_slow[i]) or np.isnan(rsi_1d_aligned[i]) or np.isnan(avg_volume[i]):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: opposite HMA crossover or volume drops below average
            if hma_fast[i] < hma_slow[i] or volume[i] < avg_volume[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: opposite HMA crossover or volume drops below average
            if hma_fast[i] > hma_slow[i] or volume[i] < avg_volume[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Volume confirmation: current volume > 1.5x average volume
            volume_ok = volume[i] > 1.5 * avg_volume[i]
            
            # Check for RSI divergence on 1D timeframe (need to look back far enough)
            if i >= 20:  # Need enough history for divergence check
                # Get recent price and RSI values for divergence detection
                lookback = 5
                start_look = max(0, i - lookback)
                price_window = close[start_look:i+1]
                # For RSI, we need to get the aligned values
                rsi_start = max(0, i - lookback)
                rsi_window = rsi_1d_aligned[rsi_start:i+1]
                
                if len(price_window) >= lookback and len(rsi_window) >= lookback and not np.any(np.isnan(rsi_window)):
                    divergence = find_divergence(price_window, rsi_window, lookback)
                    
                    # HMA crossover signals
                    hma_cross_up = hma_fast[i] > hma_slow[i] and hma_fast[i-1] <= hma_slow[i-1]
                    hma_cross_down = hma_fast[i] < hma_slow[i] and hma_fast[i-1] >= hma_slow[i-1]
                    
                    # Long entry: HMA bullish cross + bullish RSI divergence + volume
                    if hma_cross_up and divergence == 1 and volume_ok:
                        position = 1
                        signals[i] = 0.25
                    # Short entry: HMA bearish cross + bearish RSI divergence + volume
                    elif hma_cross_down and divergence == -1 and volume_ok:
                        position = -1
                        signals[i] = -0.25
    
    return signals