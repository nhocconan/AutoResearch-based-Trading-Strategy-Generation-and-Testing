#!/usr/bin/env python3
"""
1d Donchian(20) breakout with 1w EMA200 trend filter and volume confirmation
Hypothesis: Donchian breakouts on daily chart capture major trends. Weekly EMA200 filters for primary trend direction (bull/bear). Volume surge confirms breakout strength. Works in bull (buy breakouts above EMA200) and bear (sell breakdowns below EMA200). Target: 30-100 trades over 4 years (7-25/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_donchian20_1wtrend_vol_v1"
timeframe = "1d"
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
    
    # 1w EMA200 for trend bias
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 200:
        ema_1w[199] = np.mean(close_1w[:200])
        for i in range(200, len(close_1w)):
            ema_1w[i] = (close_1w[i] * 2 + ema_1w[i-1] * 18) / 20
    
    # Trend bias: above EMA = bullish, below = bearish
    trend_bias_1w = np.where(close_1w > ema_1w, 1, -1)
    
    # Align to 1d timeframe
    trend_bias_aligned = align_htf_to_ltf(prices, df_1w, trend_bias_1w)
    
    # Donchian channels (20-period) from 1w data
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Upper and lower bands from previous week to avoid look-ahead
    upper_1w = np.full_like(high_1w, np.nan)
    lower_1w = np.full_like(low_1w, np.nan)
    
    for i in range(20, len(high_1w)):
        upper_1w[i] = np.max(high_1w[i-20:i])
        lower_1w[i] = np.min(low_1w[i-20:i])
    
    # Align to 1d timeframe
    upper_aligned = align_htf_to_ltf(prices, df_1w, upper_1w)
    lower_aligned = align_htf_to_ltf(prices, df_1w, lower_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = 30  # Need enough data for Donchian
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(atr[i]) or np.isnan(trend_bias_aligned[i]) or 
            np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i])):
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
            # Exit: price breaks below lower Donchian OR against 1w trend
            # Stoploss: price drops 2*ATR below entry
            if (close[i] < lower_aligned[i] or
                trend_bias_aligned[i] == -1 or
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
            if (close[i] > upper_aligned[i] or
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
            # Minimum holding period: only allow new entry after 10 bars flat
            if bars_since_entry >= 10:
                # Breakout entries: upper/lower with trend
                bull_breakout = close[i] > upper_aligned[i]
                bear_breakout = close[i] < lower_aligned[i]
                
                # Long: breakout with uptrend + volume
                if bull_breakout and trend_bias_aligned[i] == 1 and volume_filter:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                # Short: breakdown with downtrend + volume
                elif bear_breakout and trend_bias_aligned[i] == -1 and volume_filter:
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