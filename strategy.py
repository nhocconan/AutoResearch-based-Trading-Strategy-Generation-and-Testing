#!/usr/bin/env python3
"""
1D Donchian(20) Breakout + Volume Spike + Weekly Trend Filter + ATR Stoploss
Hypothesis: Daily Donchian breakouts with volume surge (>2x 20-day average) and weekly trend alignment (price above/below 50-week EMA) capture strong trends. Weekly filter reduces whipsaws in sideways markets. Target: 50-100 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_donchian20_volume_weeklytrend_v1"
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
    
    # Weekly EMA 50 (using 1w data)
    df_weekly = get_htf_data(prices, '1w')
    weekly_close = df_weekly['close'].values
    weekly_ema50 = np.full(len(weekly_close), np.nan)
    if len(weekly_close) >= 50:
        alpha = 2.0 / (50 + 1)
        weekly_ema50[0] = weekly_close[0]
        for i in range(1, len(weekly_close)):
            weekly_ema50[i] = alpha * weekly_close[i] + (1 - alpha) * weekly_ema50[i-1]
    weekly_ema50_aligned = align_htf_to_ltf(prices, df_weekly, weekly_ema50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = 50  # For Donchian and weekly EMA
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(atr[i]) or np.isnan(weekly_ema50_aligned[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            bars_since_entry += 1
            continue
        
        # Donchian channel (20-period)
        highest_high = np.max(high[i-20:i])
        lowest_low = np.min(low[i-20:i])
        
        # Volume filter (20-period average)
        vol_ma = np.mean(volume[i-20:i])
        volume_filter = volume[i] > vol_ma * 2.0
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price closes below Donchian lower OR below weekly EMA50
            # Stoploss: price drops 2*ATR below entry
            if (close[i] < lowest_low or
                close[i] < weekly_ema50_aligned[i] or
                close[i] < entry_price - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.25
            bars_since_entry += 1
        elif position == -1:  # short position
            # Exit: price closes above Donchian upper OR above weekly EMA50
            # Stoploss: price rises 2*ATR above entry
            if (close[i] > highest_high or
                close[i] > weekly_ema50_aligned[i] or
                close[i] > entry_price + 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -0.25
            bars_since_entry += 1
        else:
            # Look for entries: Donchian breakout + volume + weekly trend filter
            # Minimum holding period: only allow new entry after 30 bars flat
            if bars_since_entry >= 30:
                bull_breakout = close[i] > highest_high
                bear_breakout = close[i] < lowest_low
                weekly_uptrend = close[i] > weekly_ema50_aligned[i]
                weekly_downtrend = close[i] < weekly_ema50_aligned[i]
                
                if bull_breakout and volume_filter and weekly_uptrend:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                elif bear_breakout and volume_filter and weekly_downtrend:
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