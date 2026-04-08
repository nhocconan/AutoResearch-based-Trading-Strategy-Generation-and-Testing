#!/usr/bin/env python3
"""
12h_2025_momentum_reversal_v1
Hypothesis: In 2025's bearish/ranging market, momentum reversals at extreme RSI levels with volume confirmation work well on 12h timeframe. Uses RSI(14) for overbought/oversold conditions, volume spike for confirmation, and 1-week trend filter to avoid counter-trend trades. Designed for low trade frequency (12-37/year) to minimize fee drag in choppy markets.
"""
name = "12h_2025_momentum_reversal_v1"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_rsi(close, period=14):
    """Calculate RSI with proper Wilder's smoothing"""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing: alpha = 1/period
    avg_gain = np.zeros_like(close)
    avg_loss = np.zeros_like(close)
    
    # Initial average
    avg_gain[period] = np.mean(gain[1:period+1]) if period < len(gain) else np.nan
    avg_loss[period] = np.mean(loss[1:period+1]) if period < len(loss) else np.nan
    
    # Wilder smoothing
    for i in range(period+1, len(close)):
        avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
        avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1-week trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_1w = np.where(close_1w > ema50_1w, 1, -1)
    trend_1w_aligned = align_htf_to_ltf(prices, df_1w, trend_1w)
    
    # RSI on 12h closes
    rsi = calculate_rsi(close, 14)
    
    # Volume spike detection (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    vol_spike = vol_ratio > 2.0  # Require 2x average volume for confirmation
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 50  # Wait for RSI and volume MA to stabilize
    
    for i in range(start_idx, n):
        # Skip if required data is not ready
        if np.isnan(rsi[i]) or np.isnan(vol_ma[i]) or np.isnan(trend_1w_aligned[i]):
            continue
        
        if position == 1:  # Long position
            # Exit: RSI returns to neutral territory (50) or trend changes
            if rsi[i] >= 50 or trend_1w_aligned[i] == -1:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: RSI returns to neutral territory (50) or trend changes
            if rsi[i] <= 50 or trend_1w_aligned[i] == 1:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Only trade with volume spike confirmation
            if not vol_spike[i]:
                continue
                
            # Look for extreme RSI readings with volume confirmation
            # Long when oversold (RSI < 30) in uptrend or ranging market
            # Short when overbought (RSI > 70) in downtrend or ranging market
            if rsi[i] < 30 and trend_1w_aligned[i] != -1:  # Not in strong downtrend
                position = 1
                signals[i] = 0.25
            elif rsi[i] > 70 and trend_1w_aligned[i] != 1:  # Not in strong uptrend
                position = -1
                signals[i] = -0.25
    
    return signals