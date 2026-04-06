#!/usr/bin/env python3
"""
1h RSI(14) Pullback with 4h/1d Trend Filter + Volume Spike
Hypothesis: In trending markets (4h/1d), buy pullbacks in uptrends and sell rallies in downtrends.
Uses RSI for entry timing on 1h, with 4h EMA50 and 1d EMA200 as trend filters.
Volume spike confirms institutional interest. Designed for 15-35 trades/year to avoid fee drag.
Works in bull (buy uptrend pullbacks) and bear (sell downtrend rallies).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_rsi_pullback_4h1dtrend_vol_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 14-period RSI
    rsi = np.full(n, np.nan)
    if n >= 15:
        delta = np.diff(close)
        up = np.where(delta > 0, delta, 0)
        down = np.where(delta < 0, -delta, 0)
        roll_up = pd.Series(up).ewm(alpha=1/14, adjust=False).mean()
        roll_down = pd.Series(down).ewm(alpha=1/14, adjust=False).mean()
        rs = roll_up / (roll_down + 1e-10)
        rsi[1:] = 100 - (100 / (1 + rs))
    
    # 4h EMA50 for trend
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema_4h = np.full(len(close_4h), np.nan)
    if len(close_4h) >= 50:
        ema_4h[49] = np.mean(close_4h[:50])
        for i in range(50, len(close_4h)):
            ema_4h[i] = (close_4h[i] * 2 + ema_4h[i-1] * 49) / 50
    trend_4h = np.where(close_4h > ema_4h, 1, -1)
    trend_4h_aligned = align_htf_to_ltf(prices, df_4h, trend_4h)
    
    # 1d EMA200 for stronger trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 200:
        ema_1d[199] = np.mean(close_1d[:200])
        for i in range(200, len(close_1d)):
            ema_1d[i] = (close_1d[i] * 2 + ema_1d[i-1] * 199) / 200
    trend_1d = np.where(close_1d > ema_1d, 1, -1)
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # Volume filter (20-period average)
    vol_ma = np.full(n, np.nan)
    if n >= 20:
        vol_ma[19] = np.mean(volume[:20])
        for i in range(20, n):
            vol_ma[i] = (volume[i] + vol_ma[i-1] * 19) / 20
    volume_filter = volume > vol_ma * 1.5
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = max(50, 200)  # For RSI and EMA200
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(rsi[i]) or np.isnan(trend_4h_aligned[i]) or np.isnan(trend_1d_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            bars_since_entry += 1
            continue
        
        # Check exits
        if position == 1:  # long position
            # Exit: RSI > 70 (overbought) OR against 4h trend OR against 1d trend
            if (rsi[i] > 70 or
                trend_4h_aligned[i] == -1 or
                trend_1d_aligned[i] == -1):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.20
            bars_since_entry += 1
        elif position == -1:  # short position
            # Exit: RSI < 30 (oversold) OR against 4h trend OR against 1d trend
            if (rsi[i] < 30 or
                trend_4h_aligned[i] == 1 or
                trend_1d_aligned[i] == 1):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -0.20
            bars_since_entry += 1
        else:
            # Look for entries: RSI pullback + trend alignment + volume + session
            # Minimum holding period: only allow new entry after 12 bars flat
            if bars_since_entry >= 12 and session_filter[i]:
                # Long: RSI < 40 (pullback) in uptrend on both 4h and 1d
                if (rsi[i] < 40 and
                    trend_4h_aligned[i] == 1 and
                    trend_1d_aligned[i] == 1 and
                    volume_filter[i]):
                    signals[i] = 0.20
                    position = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                # Short: RSI > 60 (pullback) in downtrend on both 4h and 1d
                elif (rsi[i] > 60 and
                      trend_4h_aligned[i] == -1 and
                      trend_1d_aligned[i] == -1 and
                      volume_filter[i]):
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