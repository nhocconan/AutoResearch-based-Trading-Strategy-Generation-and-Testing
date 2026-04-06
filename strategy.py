#!/usr/bin/env python3
"""
4h Donchian(20) breakout with 1d EMA50 and 1w EMA200 trend filter
Hypothesis: 4h breakouts capture momentum with trend alignment from higher timeframes.
Long when price breaks above Donchian(20) and above 1d EMA50 and 1w EMA200.
Short when price breaks below Donchian(20) and below 1d EMA50 and 1w EMA200.
Volume confirmation ensures breakout strength. Target: 75-200 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian20_1d_1w_trend_vol_v2"
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
    
    # Get 1d data for trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # EMA50 on 1d close
    ema_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 50:
        ema_1d[49] = np.mean(close_1d[:50])
        for i in range(50, len(close_1d)):
            ema_1d[i] = (close_1d[i] * 2 + ema_1d[i-1] * 48) / 50
    
    # Get 1w data for trend filter (EMA200)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # EMA200 on 1w close
    ema_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 200:
        ema_1w[199] = np.mean(close_1w[:200])
        for i in range(200, len(close_1w)):
            ema_1w[i] = (close_1w[i] * 2 + ema_1w[i-1] * 198) / 200
    
    # 1d trend: above EMA50 = bullish, below = bearish
    trend_1d = np.where(close_1d > ema_1d, 1, -1)
    # 1w trend: above EMA200 = bullish, below = bearish
    trend_1w = np.where(close_1w > ema_1w, 1, -1)
    
    # Align trends to 4h timeframe
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    trend_1w_aligned = align_htf_to_ltf(prices, df_1w, trend_1w)
    
    # Get volume data for confirmation
    volume_1d = df_1d['volume'].values
    volume_1w = df_1w['volume'].values
    
    # 20-period average volume on 1d
    vol_ma_1d = np.full(len(volume_1d), np.nan)
    for i in range(20, len(volume_1d)):
        vol_ma_1d[i] = np.mean(volume_1d[i-20:i])
    
    # 20-period average volume on 1w
    vol_ma_1w = np.full(len(volume_1w), np.nan)
    for i in range(20, len(volume_1w)):
        vol_ma_1w[i] = np.mean(volume_1w[i-20:i])
    
    # Align volume MA to 4h timeframe
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    vol_ma_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_1w)
    
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
        if (np.isnan(atr[i]) or np.isnan(trend_1d_aligned[i]) or np.isnan(trend_1w_aligned[i]) or 
            np.isnan(upper[i]) or np.isnan(lower[i]) or
            np.isnan(vol_ma_1d_aligned[i]) or np.isnan(vol_ma_1w_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            bars_since_entry += 1
            continue
        
        # Volume filter: current 4h volume > 1.3x average of 1d and 1w volume (scaled)
        # Scale 1d volume to 4h: approx 1/6 of 1d volume (since 6x 4h in 1d)
        # Scale 1w volume to 4h: approx 1/42 of 1w volume (since 42x 4h in 1w)
        vol_threshold = (vol_ma_1d_aligned[i] / 6.0 + vol_ma_1w_aligned[i] / 42.0) / 2.0 * 1.3
        volume_filter = volume[i] > vol_threshold
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price breaks below lower Donchian OR against trend
            # Stoploss: price drops 2*ATR below entry
            if (close[i] < lower[i] or
                trend_1d_aligned[i] == -1 or
                trend_1w_aligned[i] == -1 or
                close[i] < entry_price - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.25
            bars_since_entry += 1
        elif position == -1:  # short position
            # Exit: price breaks above upper Donchian OR against trend
            # Stoploss: price rises 2*ATR above entry
            if (close[i] > upper[i] or
                trend_1d_aligned[i] == 1 or
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
            # Minimum holding period: only allow new entry after 6 bars flat
            if bars_since_entry >= 6:
                # Breakout entries: upper/lower with trend
                bull_breakout = close[i] > upper[i]
                bear_breakout = close[i] < lower[i]
                
                # Long: breakout above upper with bullish trend + volume
                if bull_breakout and trend_1d_aligned[i] == 1 and trend_1w_aligned[i] == 1 and volume_filter:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                # Short: breakdown below lower with bearish trend + volume
                elif bear_breakout and trend_1d_aligned[i] == -1 and trend_1w_aligned[i] == -1 and volume_filter:
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