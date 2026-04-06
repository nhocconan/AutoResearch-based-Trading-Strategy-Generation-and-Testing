#!/usr/bin/env python3
"""
4h Weekly Pivot + Volume Confirmation + ATR Stop
Hypothesis: Uses weekly pivot levels (R4/S4 for continuation breakouts, R3/S3 for mean reversion)
with volume confirmation to capture institutional flow. Aligns with 1d trend filter to avoid counter-trend trades.
Designed for 4h timeframe to target 75-200 trades over 4 years (19-50/year) to minimize fee drag.
Works in bull (breakouts with trend) and bear (breakdowns with trend).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_weeklypivot_volume_v1"
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
    
    # 1d EMA50 for trend bias
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 50:
        ema_1d[49] = np.mean(close_1d[:50])
        for i in range(50, len(close_1d)):
            ema_1d[i] = (close_1d[i] * 2 + ema_1d[i-1] * 18) / 20
    
    # Trend bias: above EMA = bullish, below = bearish
    trend_bias_1d = np.where(close_1d > ema_1d, 1, -1)
    
    # Align to 4h timeframe
    trend_bias_aligned = align_htf_to_ltf(prices, df_1d, trend_bias_1d)
    
    # Calculate weekly pivot from 1d data (using previous week's data)
    # Approximate weekly using last 5 days
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Weekly high/low/close (using previous 5 days)
    weekly_high = np.full_like(close_1d, np.nan)
    weekly_low = np.full_like(close_1d, np.nan)
    weekly_close = np.full_like(close_1d, np.nan)
    
    for i in range(5, len(close_1d)):
        weekly_high[i] = np.max(high_1d[i-5:i])
        weekly_low[i] = np.min(low_1d[i-5:i])
        weekly_close[i] = close_1d[i-1]  # Previous day's close
    
    # Calculate weekly pivot levels
    pivot_w = np.full_like(close_1d, np.nan)
    r1_w = np.full_like(close_1d, np.nan)
    s1_w = np.full_like(close_1d, np.nan)
    r2_w = np.full_like(close_1d, np.nan)
    s2_w = np.full_like(close_1d, np.nan)
    r3_w = np.full_like(close_1d, np.nan)
    s3_w = np.full_like(close_1d, np.nan)
    r4_w = np.full_like(close_1d, np.nan)
    s4_w = np.full_like(close_1d, np.nan)
    
    for i in range(5, len(close_1d)):
        wh = weekly_high[i]
        wl = weekly_low[i]
        wc = weekly_close[i]
        
        if not (np.isnan(wh) or np.isnan(wl) or np.isnan(wc)):
            p = (wh + wl + wc) / 3.0
            pivot_w[i] = p
            r1_w[i] = 2*p - wl
            s1_w[i] = 2*p - wh
            r2_w[i] = p + (wh - wl)
            s2_w[i] = p - (wh - wl)
            r3_w[i] = wh + 2*(p - wl)
            s3_w[i] = wl - 2*(wh - p)
            r4_w[i] = 3*p - 2*wl
            s4_w[i] = 3*wh - 2*wl
    
    # Align weekly pivot levels to 4h timeframe
    pivot_w_aligned = align_htf_to_ltf(prices, df_1d, pivot_w)
    r1_w_aligned = align_htf_to_ltf(prices, df_1d, r1_w)
    s1_w_aligned = align_htf_to_ltf(prices, df_1d, s1_w)
    r2_w_aligned = align_htf_to_ltf(prices, df_1d, r2_w)
    s2_w_aligned = align_htf_to_ltf(prices, df_1d, s2_w)
    r3_w_aligned = align_htf_to_ltf(prices, df_1d, r3_w)
    s3_w_aligned = align_htf_to_ltf(prices, df_1d, s3_w)
    r4_w_aligned = align_htf_to_ltf(prices, df_1d, r4_w)
    s4_w_aligned = align_htf_to_ltf(prices, df_1d, s4_w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = 20  # Need enough data for calculations
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(atr[i]) or np.isnan(trend_bias_aligned[i]) or 
            np.isnan(r4_w_aligned[i]) or np.isnan(s4_w_aligned[i]) or
            np.isnan(r3_w_aligned[i]) or np.isnan(s3_w_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            bars_since_entry += 1
            continue
        
        # Volume filter (20-period average)
        vol_ma = np.mean(volume[max(0, i-20):i])
        volume_filter = volume[i] > vol_ma * 1.5
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price drops below S3 (mean reversion) OR against 1d trend
            # Stoploss: price drops 2*ATR below entry
            if (close[i] < s3_w_aligned[i] or
                trend_bias_aligned[i] == -1 or
                close[i] < entry_price - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.25
            bars_since_entry += 1
        elif position == -1:  # short position
            # Exit: price rises above R3 (mean reversion) OR against 1d trend
            # Stoploss: price rises 2*ATR above entry
            if (close[i] > r3_w_aligned[i] or
                trend_bias_aligned[i] == 1 or
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
                # Breakout entries: R4/S4 with trend
                bull_breakout = close[i] > r4_w_aligned[i]
                bear_breakout = close[i] < s4_w_aligned[i]
                
                # Mean reversion entries: R3/S3 counter-trend (fade)
                # Only in ranging markets - we'll use proximity to pivot as proxy
                near_pivot = abs(close[i] - pivot_w_aligned[i]) < (r1_w_aligned[i] - s1_w_aligned[i]) * 0.5
                
                # Long: breakout with trend OR mean reversion at S3 with volume
                if (bull_breakout and trend_bias_aligned[i] == 1 and volume_filter) or \
                   (close[i] > s3_w_aligned[i] and close[i] < pivot_w_aligned[i] and 
                    near_pivot and volume_filter and trend_bias_aligned[i] == -1):
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                # Short: breakdown with trend OR mean reversion at R3 with volume
                elif (bear_breakout and trend_bias_aligned[i] == -1 and volume_filter) or \
                     (close[i] < r3_w_aligned[i] and close[i] > pivot_w_aligned[i] and 
                      near_pivot and volume_filter and trend_bias_aligned[i] == 1):
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