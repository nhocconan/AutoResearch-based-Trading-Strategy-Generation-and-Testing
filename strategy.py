#!/usr/bin/env python3
"""
12h_MultiTimeframe_Trend_Scalper
Hypothesis: Combines daily trend filter (EMA50) with 12h price action (breakout of prior swing high/low) and volume confirmation. Uses tight risk management to limit trades and avoid overtrading. Works in both bull and bear by following higher timeframe trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = np.zeros_like(close_1d)
    ema_50_1d[:] = np.nan
    if len(close_1d) >= 50:
        k = 2 / (50 + 1)
        ema_50_1d[49] = np.mean(close_1d[:50])
        for i in range(50, len(close_1d)):
            ema_50_1d[i] = close_1d[i] * k + ema_50_1d[i-1] * (1 - k)
    
    # Align daily EMA50 to 12h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 12h swing points (pivots) for breakout levels
    swing_high = np.zeros(n)
    swing_low = np.zeros(n)
    swing_high[:] = np.nan
    swing_low[:] = np.nan
    
    # Find swing highs and lows using 5-bar lookback
    for i in range(2, n-2):
        # Swing high: higher than 2 bars on each side
        if (high[i] > high[i-1] and high[i] > high[i-2] and 
            high[i] > high[i+1] and high[i] > high[i+2]):
            swing_high[i] = high[i]
        # Swing low: lower than 2 bars on each side
        if (low[i] < low[i-1] and low[i] < low[i-2] and 
            low[i] < low[i+1] and low[i] < low[i+2]):
            swing_low[i] = low[i]
    
    # Forward fill swing levels to use as breakout triggers
    for i in range(1, n):
        if np.isnan(swing_high[i]):
            swing_high[i] = swing_high[i-1]
        if np.isnan(swing_low[i]):
            swing_low[i] = swing_low[i-1]
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = np.zeros(n)
    for i in range(n):
        if i < 20:
            vol_ma[i] = np.mean(volume[0:i+1]) if i >= 0 else volume[i]
        else:
            vol_ma[i] = np.mean(volume[i-20+1:i+1])
    vol_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0
    
    start_idx = 50  # Warmup for daily EMA
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_aligned[i]) or np.isnan(swing_high[i]) or 
            np.isnan(swing_low[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        bars_since_entry += 1
        
        if position == 0:
            # Long: price breaks above prior swing high with volume spike and above daily EMA50
            if (close[i] > swing_high[i] and vol_spike[i] and 
                close[i] > ema_50_aligned[i]):
                signals[i] = 0.25
                position = 1
                bars_since_entry = 0
            # Short: price breaks below prior swing low with volume spike and below daily EMA50
            elif (close[i] < swing_low[i] and vol_spike[i] and 
                  close[i] < ema_50_aligned[i]):
                signals[i] = -0.25
                position = -1
                bars_since_entry = 0
        
        elif position == 1:
            # Exit: stop loss (2x ATR), or reversal signal
            # Calculate ATR for stop loss
            if i >= 14:
                tr1 = high[i] - low[i]
                tr2 = abs(high[i] - close[i-1])
                tr3 = abs(low[i] - close[i-1])
                tr = max(tr1, tr2, tr3)
                # Simple ATR approximation using recent TR
                atr = np.mean([
                    max(high[i-14] - low[i-14], abs(high[i-14] - close[i-15]), abs(low[i-14] - close[i-15])),
                    max(high[i-7] - low[i-7], abs(high[i-7] - close[i-8]), abs(low[i-7] - close[i-8]))
                ]) if i >= 15 else tr
            else:
                atr = (high[i] - low[i]) * 0.1  # fallback
            
            stop_price = close[i] - 2.0 * atr if i > 0 else close[i]
            
            if (bars_since_entry >= 2 and 
                (close[i] < stop_price or 
                 close[i] < swing_low[i] or 
                 close[i] < ema_50_aligned[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: stop loss (2x ATR), or reversal signal
            if i >= 14:
                tr1 = high[i] - low[i]
                tr2 = abs(high[i] - close[i-1])
                tr3 = abs(low[i] - close[i-1])
                tr = max(tr1, tr2, tr3)
                atr = np.mean([
                    max(high[i-14] - low[i-14], abs(high[i-14] - close[i-15]), abs(low[i-14] - close[i-15])),
                    max(high[i-7] - low[i-7], abs(high[i-7] - close[i-8]), abs(low[i-7] - close[i-8]))
                ]) if i >= 15 else tr
            else:
                atr = (high[i] - low[i]) * 0.1
            
            stop_price = close[i] + 2.0 * atr if i > 0 else close[i]
            
            if (bars_since_entry >= 2 and 
                (close[i] > stop_price or 
                 close[i] > swing_high[i] or 
                 close[i] > ema_50_aligned[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_MultiTimeframe_Trend_Scalper"
timeframe = "12h"
leverage = 1.0