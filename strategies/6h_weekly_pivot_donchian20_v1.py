#!/usr/bin/env python3
"""
4h Donchian(20) breakout with 12h volume confirmation and 12h trend filter
Hypothesis: Donchian breakouts capture institutional momentum, filtered by 12h EMA trend for bias and 12h volume for conviction. Works in bull (buy breakouts above 12h EMA) and bear (sell breakdowns below 12h EMA). Target: 75-200 total trades over 4 years (19-50/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian20_12h_trend_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
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
    
    # Get 12h data for trend filter (EMA21)
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # EMA21 on 12h close
    ema_12h = np.full(len(close_12h), np.nan)
    if len(close_12h) >= 21:
        ema_12h[20] = np.mean(close_12h[:21])
        for i in range(21, len(close_12h)):
            ema_12h[i] = (close_12h[i] * 2 + ema_12h[i-1] * 19) / 21
    
    # 12h trend: above EMA21 = bullish, below = bearish
    trend_12h = np.where(close_12h > ema_12h, 1, -1)
    
    # Align 12h trend to 4h timeframe
    trend_12h_aligned = align_htf_to_ltf(prices, df_12h, trend_12h)
    
    # Get 12h data for volume confirmation
    volume_12h = df_12h['volume'].values
    
    # 20-period average volume on 12h
    vol_ma_12h = np.full(len(volume_12h), np.nan)
    for i in range(20, len(volume_12h)):
        vol_ma_12h[i] = np.mean(volume_12h[i-20:i])
    
    # Align volume MA to 4h timeframe
    vol_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_12h)
    
    # Donchian channels (20-period) from 4h data
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
        if (np.isnan(atr[i]) or np.isnan(trend_12h_aligned[i]) or 
            np.isnan(upper[i]) or np.isnan(lower[i]) or
            np.isnan(vol_ma_12h_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            bars_since_entry += 1
            continue
        
        # Volume filter: current 4h volume > 1.5x 12h average volume (scaled)
        # Scale 12h volume to 4h: approx 1/3 of 12h volume (since 3x 4h in 12h)
        vol_threshold = vol_ma_12h_aligned[i] / 3.0 * 1.5
        volume_filter = volume[i] > vol_threshold
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price breaks below lower Donchian OR against 12h trend
            # Stoploss: price drops 2*ATR below entry
            if (close[i] < lower[i] or
                trend_12h_aligned[i] == -1 or
                close[i] < entry_price - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.25
            bars_since_entry += 1
        elif position == -1:  # short position
            # Exit: price breaks above upper Donchian OR against 12h trend
            # Stoploss: price rises 2*ATR above entry
            if (close[i] > upper[i] or
                trend_12h_aligned[i] == 1 or
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
                # Breakout entries: upper/lower with 12h trend
                bull_breakout = close[i] > upper[i]
                bear_breakout = close[i] < lower[i]
                
                # Long: breakout above upper with bullish 12h trend + volume
                if bull_breakout and trend_12h_aligned[i] == 1 and volume_filter:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                # Short: breakdown below lower with bearish 12h trend + volume
                elif bear_breakout and trend_12h_aligned[i] == -1 and volume_filter:
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