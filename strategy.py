#!/usr/bin/env python3
"""
6h_RSI_Divergence_Pattern_1wTrend
Hypothesis: Detect RSI divergence patterns on 6h timeframe with weekly trend filter to capture reversals in both bull and bear markets.
Uses RSI(14) for momentum and weekly EMA(50) for trend direction. Looks for bullish/bearish divergence where price makes new high/low but RSI does not.
Adds volume confirmation to reduce false signals. Designed for low trade frequency (<30/year) to minimize fee drag.
"""

name = "6h_RSI_Divergence_Pattern_1wTrend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA50 for trend filter
    ema_50_1w = np.full_like(close_1w, np.nan)
    if len(close_1w) >= 50:
        ema_50_1w[49] = np.mean(close_1w[0:50])
        for i in range(50, len(close_1w)):
            ema_50_1w[i] = ema_50_1w[i-1] * 0.9607843137 + close_1w[i] * 0.0392156863  # EMA alpha = 2/(50+1)
    
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate RSI(14) on 6h data
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing (alpha = 1/14)
    alpha = 1.0 / 14
    avg_gain = np.full_like(gain, np.nan)
    avg_loss = np.full_like(loss, np.nan)
    
    if len(gain) >= 14:
        avg_gain[13] = np.mean(gain[0:14])
        avg_loss[13] = np.mean(loss[0:14])
        for i in range(14, len(gain)):
            avg_gain[i] = alpha * gain[i] + (1 - alpha) * avg_gain[i-1]
            avg_loss[i] = alpha * loss[i] + (1 - alpha) * avg_loss[i-1]
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume spike filter: current volume / 20-period average volume
    vol_ma = np.full_like(volume, np.nan)
    if len(volume) >= 20:
        vol_ma[19] = np.mean(volume[0:20])
        for i in range(20, len(volume)):
            vol_ma[i] = (vol_ma[i-1] * 19 + volume[i]) / 20
    
    volume_ratio = np.full_like(volume, np.nan)
    valid = (~np.isnan(vol_ma)) & (vol_ma != 0)
    volume_ratio[valid] = volume[valid] / vol_ma[valid]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0
    
    start_idx = max(30, 50)  # Ensure RSI and volume MA are ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(rsi[i]) or np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(volume_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        bars_since_entry += 1
        
        if position == 0:
            # Look for bullish divergence: price makes new low, RSI makes higher low
            # Bearish divergence: price makes new high, RSI makes lower high
            lookback = 10
            if i >= lookback:
                price_low = np.min(low[i-lookback:i+1])
                price_high = np.max(high[i-lookback:i+1])
                rsi_low = np.min(rsi[i-lookback:i+1])
                rsi_high = np.max(rsi[i-lookback:i+1])
                
                # Current price at recent extremes
                is_at_low = low[i] <= price_low * 1.001  # within 0.1% of recent low
                is_at_high = high[i] >= price_high * 0.999  # within 0.1% of recent high
                
                bullish_div = is_at_low and (rsi[i] > rsi_low + 5) and (rsi[i-lookback:i+1].argmin() == 0)
                bearish_div = is_at_high and (rsi[i] < rsi_high - 5) and (rsi[i-lookback:i+1].argmax() == 0)
                
                # Enter long on bullish divergence in uptrend (price > weekly EMA50)
                if bullish_div and close[i] > ema_50_1w_aligned[i] and volume_ratio[i] > 1.5:
                    signals[i] = 0.25
                    position = 1
                    bars_since_entry = 0
                # Enter short on bearish divergence in downtrend (price < weekly EMA50)
                elif bearish_div and close[i] < ema_50_1w_aligned[i] and volume_ratio[i] > 1.5:
                    signals[i] = -0.25
                    position = -1
                    bars_since_entry = 0
        
        elif position == 1:
            # Exit conditions: RSI overbought or trend reversal
            if rsi[i] > 70 or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit conditions: RSI oversold or trend reversal
            if rsi[i] < 30 or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -0.25
    
    return signals