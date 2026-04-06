#!/usr/bin/env python3
"""
6h Donchian(20) Breakout + 12h Volume Confirmation + Trend Filter
Hypothesis: Combines price breakout with volume confirmation and trend alignment.
Uses 6h as primary timeframe with 12h for volume and trend filters to reduce false signals.
Designed to work in both bull and bear markets by requiring volume surge and trend alignment.
Target: 50-150 total trades over 4 years with strict entry criteria to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_donchian20_12hvol_trend_v1"
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
    
    # 14-period ATR for stoploss
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
    
    # Load 12h data once for volume and trend filters
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # 12h EMA(20) for trend filter
    ema_12h = np.full(len(close_12h), np.nan)
    if len(close_12h) >= 20:
        ema_12h[19] = np.mean(close_12h[:20])
        for i in range(20, len(close_12h)):
            ema_12h[i] = (close_12h[i] * 2 + ema_12h[i-1] * 18) / 20
    
    # 12h volume average (20-period)
    vol_ma_12h = np.full(len(volume_12h), np.nan)
    if len(volume_12h) >= 20:
        for i in range(19, len(volume_12h)):
            vol_ma_12h[i] = np.mean(volume_12h[i-19:i+1])
    
    # Align 12h indicators to 6h timeframe
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    vol_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_exit = 0  # Enforce minimum holding period
    
    # Start from warmup period (need 20 bars for Donchian)
    start = 20
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(atr[i]) or np.isnan(ema_12h_aligned[i]) or np.isnan(vol_ma_12h_aligned[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            bars_since_exit += 1
            continue
        
        # Donchian channel (20-period) on 6h
        highest_high = np.max(high[i-20:i])
        lowest_low = np.min(low[i-20:i])
        
        # Volume filter: current 6h volume > 1.5x 12h volume average
        volume_filter = volume[i] > vol_ma_12h_aligned[i] * 1.5
        
        # Trend filter: price relative to 12h EMA
        trend_filter_long = close[i] > ema_12h_aligned[i]
        trend_filter_short = close[i] < ema_12h_aligned[i]
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price closes below Donchian lower OR stoploss hit
            if (close[i] < lowest_low or
                close[i] < entry_price - 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            else:
                signals[i] = 0.25
            bars_since_exit += 1
        elif position == -1:  # short position
            # Exit: price closes above Donchian upper OR stoploss hit
            if (close[i] > highest_high or
                close[i] > entry_price + 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            else:
                signals[i] = -0.25
            bars_since_exit += 1
        else:
            # Look for entries: Donchian breakout + volume + trend + minimum bars since exit
            if bars_since_exit >= 10:  # Minimum 10 bars between trades
                bull_breakout = close[i] > highest_high
                bear_breakout = close[i] < lowest_low
                
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