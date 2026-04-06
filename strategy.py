#!/usr/bin/env python3
"""
1h Donchian(20) breakout with 4h EMA20 trend and volume confirmation
Hypothesis: 1h breakouts capture momentum with 4h trend filter to reduce false signals.
Volume confirmation ensures breakout strength. Session filter (08-20 UTC) reduces noise.
Works in bull (buy breakouts above 4h EMA20) and bear (sell breakdowns below 4h EMA20).
Target: 60-150 total trades over 4 years (15-37/year) for 1h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_donchian20_4h_ema_vol_v2"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 40:
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
    
    # Get 4h data for trend filter (EMA20)
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # EMA20 on 4h close
    ema_4h = np.full(len(close_4h), np.nan)
    if len(close_4h) >= 20:
        ema_4h[19] = np.mean(close_4h[:20])
        for i in range(20, len(close_4h)):
            ema_4h[i] = (close_4h[i] * 2 + ema_4h[i-1] * 18) / 20
    
    # 4h trend: above EMA20 = bullish, below = bearish
    trend_4h = np.where(close_4h > ema_4h, 1, -1)
    
    # Align trend to 1h timeframe
    trend_4h_aligned = align_htf_to_ltf(prices, df_4h, trend_4h)
    
    # Get 4h volume data for confirmation
    volume_4h = df_4h['volume'].values
    
    # 20-period average volume on 4h
    vol_ma_4h = np.full(len(volume_4h), np.nan)
    for i in range(20, len(volume_4h)):
        vol_ma_4h[i] = np.mean(volume_4h[i-20:i])
    
    # Align volume MA to 1h timeframe
    vol_ma_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_4h)
    
    # Donchian channels (20-period) from 1h data
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(20, n):
        upper[i] = np.max(high[i-20:i])
        lower[i] = np.min(low[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Precompute hour for session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    # Start from warmup period
    start = 40  # Need enough data for Donchian and alignments
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(atr[i]) or np.isnan(trend_4h_aligned[i]) or 
            np.isnan(upper[i]) or np.isnan(lower[i]) or
            np.isnan(vol_ma_4h_aligned[i])):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            bars_since_entry += 1
            continue
        
        # Volume filter: current 1h volume > 1.5x 4h average volume (scaled)
        # Scale 4h volume to 1h: approx 1/4 of 4h volume (since 4x 1h in 4h)
        vol_threshold = vol_ma_4h_aligned[i] / 4.0 * 1.5
        volume_filter = volume[i] > vol_threshold
        
        # Session filter: 08-20 UTC
        session_filter = 8 <= hours[i] <= 20
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price breaks below lower Donchian OR against trend
            # Stoploss: price drops 2*ATR below entry
            if (close[i] < lower[i] or
                trend_4h_aligned[i] == -1 or
                close[i] < entry_price - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.20
            bars_since_entry += 1
        elif position == -1:  # short position
            # Exit: price breaks above upper Donchian OR against trend
            # Stoploss: price rises 2*ATR above entry
            if (close[i] > upper[i] or
                trend_4h_aligned[i] == 1 or
                close[i] > entry_price + 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -0.20
            bars_since_entry += 1
        else:
            # Look for entries
            # Minimum holding period: only allow new entry after 4 bars flat
            if bars_since_entry >= 4:
                # Breakout entries: upper/lower with trend
                bull_breakout = close[i] > upper[i]
                bear_breakout = close[i] < lower[i]
                
                # Long: breakout above upper with bullish trend + volume + session
                if bull_breakout and trend_4h_aligned[i] == 1 and volume_filter and session_filter:
                    signals[i] = 0.20
                    position = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                # Short: breakdown below lower with bearish trend + volume + session
                elif bear_breakout and trend_4h_aligned[i] == -1 and volume_filter and session_filter:
                    signals[i] = -0.20
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