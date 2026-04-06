#!/usr/bin/env python3
"""
12h Donchian(20) breakout with 1w EMA200 trend filter and volume confirmation
Hypothesis: Donchian breakouts on 12h timeframe capture momentum in trending markets. 
Filter by 1w EMA200 for trend bias (bullish when above, bearish when below) and volume confirmation for conviction.
Uses 12h timeframe to reduce trade frequency and avoid fee drag. Works in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_donchian20_1w_trend_vol_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 14-period ATR
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
    
    # Get 1w data for trend filter (EMA200)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # EMA200 on 1w close
    ema_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 200:
        ema_1w[199] = np.mean(close_1w[:200])
        for i in range(200, len(close_1w)):
            ema_1w[i] = (close_1w[i] * 2 + ema_1w[i-1] * 198) / 200
    
    # 1w trend: above EMA200 = bullish, below = bearish
    trend_1w = np.where(close_1w > ema_1w, 1, -1)
    
    # Align 1w trend to 12h timeframe
    trend_1w_aligned = align_htf_to_ltf(prices, df_1w, trend_1w)
    
    # Get 1w data for volume confirmation
    volume_1w = df_1w['volume'].values
    
    # 50-period average volume on 1w
    vol_ma_1w = np.full(len(volume_1w), np.nan)
    for i in range(50, len(volume_1w)):
        vol_ma_1w[i] = np.mean(volume_1w[i-50:i])
    
    # Align volume MA to 12h timeframe
    vol_ma_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_1w)
    
    # Donchian channels (20-period) from 12h data
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(20, n):
        upper[i] = np.max(high[i-20:i])
        lower[i] = np.min(low[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = 40  # Need enough data for Donchian and alignments
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(atr[i]) or np.isnan(trend_1w_aligned[i]) or 
            np.isnan(upper[i]) or np.isnan(lower[i]) or
            np.isnan(vol_ma_1w_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            bars_since_entry += 1
            continue
        
        # Volume filter: current 12h volume > 1.5x 1w average volume (scaled)
        # Scale 1w volume to 12h: approx 1/14 of 1w volume (since 14x 12h in 1w)
        vol_threshold = vol_ma_1w_aligned[i] / 14.0 * 1.5
        volume_filter = volume[i] > vol_threshold
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price breaks below lower Donchian OR against 1w trend
            # Stoploss: price drops 2*ATR below entry
            if (close[i] < lower[i] or
                trend_1w_aligned[i] == -1 or
                close[i] < entry_price - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.25
            bars_since_entry += 1
        elif position == -1:  # short position
            # Exit: price breaks above upper Donchian OR against 1w trend
            # Stoploss: price rises 2*ATR above entry
            if (close[i] > upper[i] or
                trend_1w_aligned[i] == 1 or
                close[i] > entry_price + 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -0.25
            bars_since_entry += 1
        else:
            # Look for entries
            # Minimum holding period: only allow new entry after 12 bars flat
            if bars_since_entry >= 12:
                # Breakout entries: upper/lower with 1w trend
                bull_breakout = close[i] > upper[i]
                bear_breakout = close[i] < lower[i]
                
                # Long: breakout above upper with bullish 1w trend + volume
                if bull_breakout and trend_1w_aligned[i] == 1 and volume_filter:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                # Short: breakdown below lower with bearish 1w trend + volume
                elif bear_breakout and trend_1w_aligned[i] == -1 and volume_filter:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
                    bars_since_entry = 0
                else:
                    signals[i] = 0.0
                    bars_since_entry += 1
            else:
                signals[i] = 0.0
                bars_since_entry += 1
    
    return signals