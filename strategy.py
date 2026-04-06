#!/usr/bin/env python3
"""
6h Donchian(20) Breakout + 1d Williams %R + Volume Filter + ATR Stoploss
Hypothesis: Donchian breakouts capture momentum; Williams %R identifies overbought/oversold conditions on daily chart to avoid buying tops or selling bottoms; volume confirms breakout strength; ATR stop limits downside. Designed to work in both bull and bear markets by using daily Williams %R as a filter to trade with intermediate-term momentum while avoiding extremes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_ltf_to_htf

name = "6h_donchian20_williamsr_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 14-period ATR for stop loss
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
    
    # Load 1d data once for Williams %R
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams %R (14-period) on daily: (Highest High - Close) / (Highest High - Lowest Low) * -100
    williams_r = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 14:
        for i in range(13, len(close_1d)):
            highest_high = np.max(high_1d[i-13:i+1])
            lowest_low = np.min(low_1d[i-13:i+1])
            if highest_high != lowest_low:
                williams_r[i] = (highest_high - close_1d[i]) / (highest_high - lowest_low) * -100
            else:
                williams_r[i] = -50  # neutral if no range
    
    # Align Williams %R to 6h timeframe
    williams_r_aligned = align_ltf_to_htf(prices, df_1d, williams_r)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_exit = 0  # bars since last exit to enforce minimum holding
    
    # Start from warmup period
    start = 20  # For Donchian
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(atr[i]) or np.isnan(williams_r_aligned[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            bars_since_exit += 1
            continue
        
        # Donchian channel (20-period)
        highest_high = np.max(high[i-20:i])
        lowest_low = np.min(low[i-20:i])
        
        # Volume filter (20-period average)
        vol_ma = np.mean(volume[i-20:i])
        volume_filter = volume[i] > vol_ma * 1.5
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price closes below Donchian lower OR Williams %R > -20 (overbought)
            # Stoploss: price drops 2*ATR below entry
            if (close[i] < lowest_low or
                williams_r_aligned[i] > -20 or
                close[i] < entry_price - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            else:
                signals[i] = 0.25
            bars_since_exit += 1
        elif position == -1:  # short position
            # Exit: price closes above Donchian upper OR Williams %R < -80 (oversold)
            # Stoploss: price rises 2*ATR above entry
            if (close[i] > highest_high or
                williams_r_aligned[i] < -80 or
                close[i] > entry_price + 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            else:
                signals[i] = -0.25
            bars_since_exit += 1
        else:
            # Look for entries: Donchian breakout + volume + Williams %R filter
            # Minimum holding period: only allow new entry after 10 bars flat
            if bars_since_exit >= 10:
                bull_breakout = close[i] > highest_high
                bear_breakout = close[i] < lowest_low
                
                # Williams %R filter: avoid overbought/oversold extremes
                # Long only when not overbought (Williams %R > -80)
                # Short only when not oversold (Williams %R < -20)
                williams_filter_long = williams_r_aligned[i] > -80
                williams_filter_short = williams_r_aligned[i] < -20
                
                if bull_breakout and volume_filter and williams_filter_long:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                    bars_since_exit = 0
                elif bear_breakout and volume_filter and williams_filter_short:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
                    bars_since_exit = 0
                else:
                    signals[i] = 0.0
                    bars_since_exit += 1
            else:
                signals[i] = 0.0
                bars_since_exit += 1
    
    return signals