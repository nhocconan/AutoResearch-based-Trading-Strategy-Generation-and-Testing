#!/usr/bin/env python3
"""
12h Camarilla pivot with 1d trend filter and volume confirmation
Hypothesis: 12h Camarilla pivot levels (S3/S4 for shorts, S1/S2 for longs) capture reversals. 
Filter by 1d EMA50 for trend bias and volume confirmation for conviction. Works in bull (buy pullbacks to S1/S2 above 1d EMA50) and bear (sell rallies to S3/S4 below 1d EMA50). 
Target: 50-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_pivot_1d_trend_vol_v1"
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
    
    # Get 1d data for trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # EMA50 on 1d close
    ema_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 50:
        ema_1d[49] = np.mean(close_1d[:50])
        for i in range(50, len(close_1d)):
            ema_1d[i] = (close_1d[i] * 2 + ema_1d[i-1] * 48) / 50
    
    # 1d trend: above EMA50 = bullish, below = bearish
    trend_1d = np.where(close_1d > ema_1d, 1, -1)
    
    # Align 1d trend to 12h timeframe
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # Get 1d data for volume confirmation
    volume_1d = df_1d['volume'].values
    
    # 20-period average volume on 1d
    vol_ma_1d = np.full(len(volume_1d), np.nan)
    for i in range(20, len(volume_1d)):
        vol_ma_1d[i] = np.mean(volume_1d[i-20:i])
    
    # Align volume MA to 12h timeframe
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Calculate Camarilla pivot levels from previous 12h bar
    # Pivot = (H + L + C) / 3
    # Range = H - L
    # S1 = C - (Range * 1.1 / 12)
    # S2 = C - (Range * 1.1 / 6)
    # S3 = C - (Range * 1.1 / 4)
    # S4 = C - (Range * 1.1 / 2)
    # R1 = C + (Range * 1.1 / 12)
    # R2 = C + (Range * 1.1 / 6)
    # R3 = C + (Range * 1.1 / 4)
    # R4 = C + (Range * 1.1 / 2)
    
    pivot = np.full(n, np.nan)
    s1 = np.full(n, np.nan)
    s2 = np.full(n, np.nan)
    s3 = np.full(n, np.nan)
    s4 = np.full(n, np.nan)
    r1 = np.full(n, np.nan)
    r2 = np.full(n, np.nan)
    r3 = np.full(n, np.nan)
    r4 = np.full(n, np.nan)
    
    for i in range(1, n):
        if not (np.isnan(high[i-1]) or np.isnan(low[i-1]) or np.isnan(close[i-1])):
            pivot[i] = (high[i-1] + low[i-1] + close[i-1]) / 3.0
            rng = high[i-1] - low[i-1]
            s1[i] = close[i-1] - (rng * 1.1 / 12)
            s2[i] = close[i-1] - (rng * 1.1 / 6)
            s3[i] = close[i-1] - (rng * 1.1 / 4)
            s4[i] = close[i-1] - (rng * 1.1 / 2)
            r1[i] = close[i-1] + (rng * 1.1 / 12)
            r2[i] = close[i-1] + (rng * 1.1 / 6)
            r3[i] = close[i-1] + (rng * 1.1 / 4)
            r4[i] = close[i-1] + (rng * 1.1 / 2)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = 50  # Need enough data for EMA and pivots
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(atr[i]) or np.isnan(trend_1d_aligned[i]) or 
            np.isnan(s1[i]) or np.isnan(s2[i]) or np.isnan(s3[i]) or np.isnan(s4[i]) or
            np.isnan(r1[i]) or np.isnan(r2[i]) or np.isnan(r3[i]) or np.isnan(r4[i]) or
            np.isnan(vol_ma_1d_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            bars_since_entry += 1
            continue
        
        # Volume filter: current 12h volume > 1.5x 1d average volume (scaled)
        # Scale 1d volume to 12h: approx 1/2 of 1d volume (since 2x 12h in 1d)
        vol_threshold = vol_ma_1d_aligned[i] / 2.0 * 1.5
        volume_filter = volume[i] > vol_threshold
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price breaks below S2 OR against 1d trend
            # Stoploss: price drops 2*ATR below entry
            if (close[i] < s2[i] or
                trend_1d_aligned[i] == -1 or
                close[i] < entry_price - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.25
            bars_since_entry += 1
        elif position == -1:  # short position
            # Exit: price breaks above R2 OR against 1d trend
            # Stoploss: price rises 2*ATR above entry
            if (close[i] > r2[i] or
                trend_1d_aligned[i] == 1 or
                close[i] > entry_price + 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -0.25
            bars_since_entry += 1
        else:
            # Look for entries
            # Minimum holding period: only allow new entry after 4 bars flat
            if bars_since_entry >= 4:
                # Long: pullback to S1/S2 with bullish 1d trend + volume
                pullback_to_support = (close[i] <= s2[i] and close[i] >= s1[i]) or \
                                    (close[i] <= s1[i] and close[i] >= s3[i])
                # Short: rally to R3/R4 with bearish 1d trend + volume
                rally_to_resistance = (close[i] >= r3[i] and close[i] <= r4[i]) or \
                                    (close[i] >= r2[i] and close[i] <= r3[i])
                
                if pullback_to_support and trend_1d_aligned[i] == 1 and volume_filter:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                elif rally_to_resistance and trend_1d_aligned[i] == -1 and volume_filter:
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