#!/usr/bin/env python3
"""
6h Donchian(20) Breakout + 12h Trend Filter + Volume Confirmation
Hypothesis: Donchian breakouts on 6h capture momentum aligned with 12h trend (via close vs SMA50), volume confirms breakout strength, and a minimum hold period prevents whipsaws. Designed for 60-120 total trades over 4 years (~15-30/year) with discrete sizing to limit fee drag. Works in both bull and bear via trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_donchian20_12h_trend_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 14-period ATR (for stoploss)
    atr = np.full(n, np.nan)
    if n >= 14:
        tr = np.maximum(
            high[1:] - low[1:],
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        )
        if len(tr) > 0:
            atr[1] = tr[0]
            for i in range(2, n):
                atr[i] = (tr[i-1] * 13 + atr[i-1]) / 14
    
    # Load 12h close once before loop for SMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    sma_12h = np.full(len(close_12h), np.nan)
    if len(close_12h) >= 50:
        sma_12h[49] = np.mean(close_12h[:50])
        for i in range(50, len(close_12h)):
            sma_12h[i] = (close_12h[i] + sma_12h[i-1] * 49) / 50
    sma_12h_aligned = align_htf_to_ltf(prices, df_12h, sma_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_exit = 0  # tracks bars since last exit to enforce min hold
    
    # Start from warmup period (max of 20 for Donchian, 50 for SMA)
    start = 50
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(atr[i]) or np.isnan(sma_12h_aligned[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            bars_since_exit += 1
            continue
        
        # Donchian channel (20-period)
        highest_high = np.max(high[i-20:i])
        lowest_low = np.min(low[i-20:i])
        
        # Volume filter (20-period average)
        vol_ma = np.mean(volume[i-20:i])
        volume_filter = volume[i] > vol_ma * 1.5
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price closes below Donchian lower or stoploss hit
            if (close[i] < lowest_low or
                close[i] < entry_price - 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            else:
                signals[i] = 0.25
            bars_since_exit += 1
        elif position == -1:  # short position
            # Exit: price closes above Donchian upper or stoploss hit
            if (close[i] > highest_high or
                close[i] > entry_price + 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            else:
                signals[i] = -0.25
            bars_since_exit += 1
        else:
            # Look for entries: Donchian breakout + volume + trend filter + min hold
            if bars_since_exit >= 20:  # minimum 20 bars (5 days) between trades
                bull_breakout = close[i] > highest_high
                bear_breakout = close[i] < lowest_low
                
                # Trend filter: long if close > 12h SMA50, short if close < 12h SMA50
                trend_filter_long = close[i] > sma_12h_aligned[i]
                trend_filter_short = close[i] < sma_12h_aligned[i]
                
                if bull_breakout and volume_filter and trend_filter_long:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                    bars_since_exit = 0
                elif bear_breakout and volume_filter and trend_filter_short:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
                    bars_since_exit = 0
                else:
                    signals[i] = 0.0
                    bars_since_exit += 1
            else:
                signals[i] = 0.0
                bars_since_exit += 1
    
    return signals