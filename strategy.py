#!/usr/bin/env python3
"""
4h Donchian(20) Breakout + 12h Supertrend Trend Filter + Volume Spike
Hypothesis: Donchian breakouts on 4h capture momentum, filtered by 12h Supertrend trend direction to avoid counter-trend trades, with volume spike confirmation for breakout strength. Uses tight entry conditions to limit trades to 75-200 over 4 years, with ATR-based stops to manage drawdown. Designed to work in both bull and bear markets by only trading in the direction of the higher timeframe trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian20_12hsupertrend_vol_v2"
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
    
    # 10-period ATR for Supertrend
    atr = np.full(n, np.nan)
    if n >= 10:
        tr = np.maximum(
            high[1:] - low[1:],
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        )
        if len(tr) > 0:
            atr[1] = tr[0]
            for i in range(2, n):
                atr[i] = (tr[i-1] * 9 + atr[i-1]) / 10
    
    # Load 12h data once before loop
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h Supertrend (10, 3.0)
    atr_12h = np.full(len(close_12h), np.nan)
    if len(close_12h) >= 10:
        tr_12h = np.maximum(
            high_12h[1:] - low_12h[1:],
            np.abs(high_12h[1:] - close_12h[:-1]),
            np.abs(low_12h[1:] - close_12h[:-1])
        )
        if len(tr_12h) > 0:
            atr_12h[1] = tr_12h[0]
            for i in range(2, len(close_12h)):
                atr_12h[i] = (tr_12h[i-1] * 9 + atr_12h[i-1]) / 10
    
    # Supertrend calculation
    supertrend = np.full(len(close_12h), np.nan)
    direction = np.full(len(close_12h), np.nan)  # 1 for uptrend, -1 for downtrend
    if len(close_12h) >= 10 and not np.all(np.isnan(atr_12h)):
        for i in range(10, len(close_12h)):
            # Basic upper and lower bands
            hl_avg = (high_12h[i] + low_12h[i]) / 2
            upper_band = hl_avg + 3.0 * atr_12h[i]
            lower_band = hl_avg - 3.0 * atr_12h[i]
            
            # Final bands
            if i == 10:
                final_upper = upper_band
                final_lower = lower_band
            else:
                final_upper = upper_band if (upper_band < supertrend[i-1] or close_12h[i-1] > supertrend[i-1]) else supertrend[i-1]
                final_lower = lower_band if (lower_band > supertrend[i-1] or close_12h[i-1] < supertrend[i-1]) else supertrend[i-1]
            
            # Supertrend and direction
            if i == 10:
                supertrend[i] = final_lower
                direction[i] = 1
            else:
                if supertrend[i-1] == final_upper[i-1]:
                    supertrend[i] = final_lower
                    direction[i] = 1
                else:
                    supertrend[i] = final_upper
                    direction[i] = -1
                
                # Reverse logic based on close
                if close_12h[i] > supertrend[i]:
                    direction[i] = 1
                else:
                    direction[i] = -1
    
    # Align Supertrend direction to 4h
    supertrend_direction_aligned = align_htf_to_ltf(prices, df_12h, direction)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_exit = 0  # bars since last exit to enforce cooldown
    
    # Start from warmup period
    start = 20  # For Donchian
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(atr[i]) or np.isnan(supertrend_direction_aligned[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            bars_since_exit += 1
            continue
        
        # Donchian channel (20-period)
        highest_high = np.max(high[i-20:i])
        lowest_low = np.min(low[i-20:i])
        
        # Volume filter (20-period average, spike > 2.0x)
        vol_ma = np.mean(volume[i-20:i])
        volume_filter = volume[i] > vol_ma * 2.0
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price closes below Donchian lower
            # Stoploss: price drops 2.5*ATR below entry
            if (close[i] < lowest_low or
                close[i] < entry_price - 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            else:
                signals[i] = 0.25
            bars_since_exit += 1
        elif position == -1:  # short position
            # Exit: price closes above Donchian upper
            # Stoploss: price rises 2.5*ATR above entry
            if (close[i] > highest_high or
                close[i] > entry_price + 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            else:
                signals[i] = -0.25
            bars_since_exit += 1
        else:
            # Look for entries: Donchian breakout + volume + trend filter
            # Minimum 10 bars cooldown between trades
            if bars_since_exit >= 10:
                bull_breakout = close[i] > highest_high
                bear_breakout = close[i] < lowest_low
                
                # Trend filter: only trade long if 12h Supertrend uptrend, short if downtrend
                trend_filter_long = supertrend_direction_aligned[i] == 1
                trend_filter_short = supertrend_direction_aligned[i] == -1
                
                if bull_breakout and volume_filter and trend_filter_long:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                    bars_since_exit = 0
                elif bear_breakout and volume_filter and trend_filter_short:
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