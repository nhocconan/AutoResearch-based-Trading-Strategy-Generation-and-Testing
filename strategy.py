#!/usr/bin/env python3
"""
4h Donchian(20) breakout with 12h EMA25 trend filter and volume confirmation
Hypothesis: Donchian breakouts capture momentum, filtered by 12h EMA for trend bias and volume for conviction. Works in bull (buy breakouts in uptrend) and bear (sell breakdowns in downtrend). Target: 80-160 trades over 4 years (20-40/year).
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
    
    # 12h EMA25 for trend bias
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_12h = np.full(len(close_12h), np.nan)
    if len(close_12h) >= 25:
        ema_12h[24] = np.mean(close_12h[:25])
        for i in range(25, len(close_12h)):
            ema_12h[i] = (close_12h[i] * 2 + ema_12h[i-1] * 18) / 20
    
    # Trend bias: above EMA = bullish, below = bearish
    trend_bias_12h = np.where(close_12h > ema_12h, 1, -1)
    
    # Align to 4h timeframe
    trend_bias_aligned = align_htf_to_ltf(prices, df_12h, trend_bias_12h)
    
    # Donchian channels (20-period) from 12h data
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Upper and lower bands from previous period to avoid look-ahead
    upper_12h = np.full_like(high_12h, np.nan)
    lower_12h = np.full_like(low_12h, np.nan)
    
    for i in range(20, len(high_12h)):
        upper_12h[i] = np.max(high_12h[i-20:i])
        lower_12h[i] = np.min(low_12h[i-20:i])
    
    # Align to 4h timeframe
    upper_aligned = align_htf_to_ltf(prices, df_12h, upper_12h)
    lower_aligned = align_htf_to_ltf(prices, df_12h, lower_12h)
    
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
            # Exit: price breaks below lower Donchian OR against 12h trend
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
            # Exit: price breaks above upper Donchian OR against 12h trend
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
            # Minimum holding period: only allow new entry after 12 bars flat
            if bars_since_entry >= 12:
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