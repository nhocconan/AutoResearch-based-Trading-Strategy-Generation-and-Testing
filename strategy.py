#!/usr/bin/env python3
"""
6h_1d_rvsi_trend_v1
Hypothesis: Use 1-day Relative Vigor Index (RVI) with 6-hour price action for trend-following entries.
- RVI measures conviction of trend by comparing close-open to high-low ranges
- Long when RVI crosses above 0.50 with price above 6h EMA20 (bullish momentum)
- Short when RVI crosses below 0.50 with price below 6h EMA20 (bearish momentum)
- Uses 1-day trend as filter to avoid counter-trend trades in choppy markets
- Designed for low trade frequency (15-30/year) to minimize fee drag
- Works in bull/bear via trend filter and momentum confirmation
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_rvsi_trend_v1"
timeframe = "6h"
leverage = 1.0

def calculate_rvi(high, low, open_, close, period=10):
    """Calculate Relative Vigor Index"""
    if len(close) < period:
        return np.full(len(close), np.nan)
    
    numerator = np.zeros(len(close))
    denominator = np.zeros(len(close))
    
    for i in range(len(close)):
        if i >= period - 1:
            # Numerator: (close - open) + 2*(close_prev - open_prev) + 2*(close_prev2 - open_prev2) + (close_prev3 - open_prev3)
            c_o = close[i] - open_[i]
            c_o1 = close[i-1] - open_[i-1] if i-1 >= 0 else 0
            c_o2 = close[i-2] - open_[i-2] if i-2 >= 0 else 0
            c_o3 = close[i-3] - open_[i-3] if i-3 >= 0 else 0
            numerator[i] = c_o + 2*c_o1 + 2*c_o2 + c_o3
            
            # Denominator: (high - low) + 2*(high_prev - low_prev) + 2*(high_prev2 - low_prev2) + (high_prev3 - low_prev3)
            h_l = high[i] - low[i]
            h_l1 = high[i-1] - low[i-1] if i-1 >= 0 else 0
            h_l2 = high[i-2] - low[i-2] if i-2 >= 0 else 0
            h_l3 = high[i-3] - low[i-3] if i-3 >= 0 else 0
            denominator[i] = h_l + 2*h_l1 + 2*h_l2 + h_l3
    
    # Smooth numerator and denominator
    num_smooth = np.zeros(len(close))
    den_smooth = np.zeros(len(close))
    
    for i in range(len(close)):
        if i >= period - 1:
            num_smooth[i] = np.mean(numerator[i-period+1:i+1])
            den_smooth[i] = np.mean(denominator[i-period+1:i+1])
    
    rvi = np.full(len(close), np.nan)
    mask = den_smooth != 0
    rvi[mask] = num_smooth[mask] / den_smooth[mask]
    
    # Signal line: 4-period SMA of RVI
    signal_line = np.full(len(close), np.nan)
    for i in range(len(close)):
        if i >= 3 and not np.isnan(rvi[i]):
            signal_line[i] = np.mean(rvi[i-3:i+1])
    
    return rvi, signal_line

def calculate_ema(close, period):
    """Calculate EMA with proper handling"""
    if len(close) < period:
        return np.full_like(close, np.nan, dtype=float)
    
    ema = np.full_like(close, np.nan, dtype=float)
    alpha = 2.0 / (period + 1)
    ema[period-1] = np.mean(close[:period])
    for i in range(period, len(close)):
        ema[i] = alpha * close[i] + (1 - alpha) * ema[i-1]
    return ema

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    open_ = prices['open'].values
    volume = prices['volume'].values
    
    # Get 1-day data for RVI trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1-day RVI for trend filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    open_1d = df_1d['open'].values
    close_1d = df_1d['close'].values
    
    rvi_1d, rvi_signal_1d = calculate_rvi(high_1d, low_1d, open_1d, close_1d, 10)
    
    # Calculate 6-hour EMA for entry timing
    ema_20_6h = calculate_ema(close, 20)
    
    # Align indicators to 6-hour timeframe
    rvi_1d_aligned = align_htf_to_ltf(prices, df_1d, rvi_1d)
    rvi_signal_1d_aligned = align_htf_to_ltf(prices, df_1d, rvi_signal_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if data not ready
        if (np.isnan(rvi_1d_aligned[i]) or np.isnan(rvi_signal_1d_aligned[i]) or
            np.isnan(ema_20_6h[i])):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        rvi = rvi_1d_aligned[i]
        rvi_signal = rvi_signal_1d_aligned[i]
        price = close[i]
        ema_20 = ema_20_6h[i]
        
        if position == 1:  # Long
            # Exit: RVI crosses below signal line OR price closes below EMA20
            if rvi < rvi_signal or price < ema_20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short
            # Exit: RVI crosses above signal line OR price closes above EMA20
            if rvi > rvi_signal or price > ema_20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: RVI crosses above signal line with price above EMA20 (bullish momentum)
            if rvi > rvi_signal and price > ema_20:
                position = 1
                signals[i] = 0.25
            # Enter short: RVI crosses below signal line with price below EMA20 (bearish momentum)
            elif rvi < rvi_signal and price < ema_20:
                position = -1
                signals[i] = -0.25
    
    return signals