#!/usr/bin/env python3
"""
4h_1d_rsi_divergence_volume_confirmation_v1
Hypothesis: 4-hour strategy using RSI divergence on daily timeframe with volume confirmation.
Looks for bullish divergence (price makes lower low, RSI makes higher low) for long entries,
and bearish divergence (price makes higher high, RSI makes lower high) for short entries.
Requires volume confirmation (>1.5x 20-period average) to filter false signals.
Designed to work in both bull and bear markets by capturing reversals at extremes.
Target: 20-35 trades/year (80-140 total over 4 years) to minimize fee drag.
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
    
    # Get daily data for RSI calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate daily RSI (14-period)
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Calculate average gain and loss
    avg_gain = np.zeros_like(close_1d)
    avg_loss = np.zeros_like(close_1d)
    
    # Initialize first average
    if len(gain) >= 14:
        avg_gain[13] = np.mean(gain[:14])
        avg_loss[13] = np.mean(loss[:14])
    
    # Calculate subsequent averages
    for i in range(14, len(close_1d)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i-1]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i-1]) / 14
    
    # Calculate RSI
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_1d = np.where(rs != 0, 100 - (100 / (1 + rs)), 100)
    rsi_1d = np.concatenate([np.full(14, np.nan), rsi_1d[14:]])
    
    # Align RSI to 4h timeframe
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Calculate ATR for volatility filter
    tr1 = np.abs(high - low)
    tr2 = np.abs(np.roll(high, 1) - close)
    tr3 = np.abs(np.roll(low, 1) - close)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(rsi_1d_aligned[i]) or np.isnan(atr[i]) or 
            i < 14):  # Need at least 14 days for RSI
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period average
        if i >= 20:
            vol_ma = np.mean(volume[max(0, i-20):i])
            volume_filter = volume[i] > vol_ma * 1.5
        else:
            volume_filter = False
        
        # Need at least 2 days of aligned data to check for divergence
        if i < 1:
            signals[i] = 0.0
            continue
            
        # Get current and previous values for divergence detection
        curr_price = close[i]
        prev_price = close[i-1]
        curr_rsi = rsi_1d_aligned[i]
        prev_rsi = rsi_1d_aligned[i-1]
        
        # Bullish divergence: price makes lower low, RSI makes higher low
        bullish_div = (curr_price < prev_price) and (curr_rsi > prev_rsi)
        # Bearish divergence: price makes higher high, RSI makes lower high
        bearish_div = (curr_price > prev_price) and (curr_rsi < prev_rsi)
        
        # Entry conditions: RSI divergence with volume confirmation
        long_entry = bullish_div and volume_filter and (curr_rsi < 40)  # Oversold condition
        short_entry = bearish_div and volume_filter and (curr_rsi > 60)  # Overbought condition
        
        # Exit conditions: RSI returns to neutral zone or opposite divergence
        long_exit = (curr_rsi > 60) or bearish_div
        short_exit = (curr_rsi < 40) or bullish_div
        
        # Position sizing: fixed 0.25 (25% of capital)
        position_size = 0.25
        
        if long_entry and position != 1:
            position = 1
            signals[i] = position_size
        elif short_entry and position != -1:
            position = -1
            signals[i] = -position_size
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_1d_rsi_divergence_volume_confirmation_v1"
timeframe = "4h"
leverage = 1.0