#!/usr/bin/env python3
"""
4h Donchian20 + 1d Trend + Volume Confirmation (Optimized)
Breakout strategy: long when price breaks above 20-period high with 1d uptrend,
short when breaks below 20-period low with 1d downtrend. Uses volume confirmation
and ATR stoploss. Optimized for 4h timeframe to target 75-200 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian20_1dtrend_vol_v1"
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
    
    # 14-period ATR with proper NaN handling
    atr = np.full(n, np.nan)
    if n >= 15:
        tr0 = np.maximum(high[0] - low[0], np.abs(high[0] - close[0]))
        atr[0] = tr0
        for i in range(1, n):
            tr = np.maximum(high[i] - low[i], np.abs(high[i] - close[i-1]), np.abs(low[i] - close[i-1]))
            atr[i] = (atr[i-1] * 13 + tr) / 14
    
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
    
    # Align trend bias to 4h timeframe
    trend_bias_aligned = align_htf_to_ltf(prices, df_1d, trend_bias_1d)
    
    # Donchian channels (20-period high/low)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    
    if n >= 20:
        # Initialize first value
        donchian_high[19] = np.max(high[0:20])
        donchian_low[19] = np.min(low[0:20])
        for i in range(20, n):
            donchian_high[i] = max(donchian_high[i-1], high[i-1])
            donchian_low[i] = min(donchian_low[i-1], low[i-1])
            # Remove the oldest value from window
            if i >= 21:
                donchian_high[i] = max(donchian_high[i], high[i-20]) if donchian_high[i] == high[i-21] else donchian_high[i]
                donchian_low[i] = min(donchian_low[i], low[i-20]) if donchian_low[i] == low[i-21] else donchian_low[i]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = 50  # Need enough data for calculations
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(atr[i]) or np.isnan(trend_bias_aligned[i]) or 
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i])):
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
            # Exit: price drops below Donchian low OR against 1d trend
            # Stoploss: price drops 2*ATR below entry
            if (close[i] < donchian_low[i] or
                trend_bias_aligned[i] == -1 or
                close[i] < entry_price - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.25
            bars_since_entry += 1
        elif position == -1:  # short position
            # Exit: price rises above Donchian high OR against 1d trend
            # Stoploss: price rises 2*ATR above entry
            if (close[i] > donchian_high[i] or
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
                # Breakout entries with trend confirmation
                bull_breakout = close[i] > donchian_high[i]
                bear_breakout = close[i] < donchian_low[i]
                
                # Long: breakout above high with uptrend + volume
                if bull_breakout and trend_bias_aligned[i] == 1 and volume_filter:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                # Short: breakout below low with downtrend + volume
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