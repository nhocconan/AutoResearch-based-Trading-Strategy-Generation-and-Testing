#!/usr/bin/env python3
"""
6h Donchian(20) Breakout + Weekly Pivot Direction + Volume Confirmation
Hypothesis: Weekly pivot levels provide structural support/resistance that works in both bull and bear markets.
Breakouts above weekly R1 or below weekly S1 with volume confirmation capture institutional interest.
Using 6h timeframe reduces trade frequency to avoid fee drag while capturing multi-day moves.
Weekly pivot filter ensures we only trade in direction of higher timeframe bias.
Target: 50-150 total trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_donchian20_weekly_pivot_vol_v1"
timeframe = "6h"
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
    
    # 14-period ATR for stoploss
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
    
    # === WEEKLY PIVOT DATA (MTF) - CALCULATE ONCE BEFORE LOOP ===
    df_weekly = get_htf_data(prices, '1w')  # Weekly data for pivot calculation
    if len(df_weekly) < 1:
        return np.zeros(n)
    
    # Calculate weekly pivot points: P = (H+L+C)/3
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    # Weekly pivot and support/resistance levels
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    weekly_r1 = 2 * weekly_pivot - weekly_low
    weekly_s1 = 2 * weekly_pivot - weekly_high
    weekly_r2 = weekly_pivot + (weekly_high - weekly_low)
    weekly_s2 = weekly_pivot - (weekly_high - weekly_low)
    weekly_r3 = weekly_high + 2 * (weekly_pivot - weekly_low)
    weekly_s3 = weekly_low - 2 * (weekly_high - weekly_pivot)
    
    # Align weekly levels to 6h timeframe (shifted by 1 week for no look-ahead)
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_weekly, weekly_pivot)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_weekly, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_weekly, weekly_s1)
    weekly_r2_aligned = align_htf_to_ltf(prices, df_weekly, weekly_r2)
    weekly_s2_aligned = align_htf_to_ltf(prices, df_weekly, weekly_s2)
    weekly_r3_aligned = align_htf_to_ltf(prices, df_weekly, weekly_r3)
    weekly_s3_aligned = align_htf_to_ltf(prices, df_weekly, weekly_s3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_exit = 0  # Bars since last exit to prevent overtrading
    
    # Start from warmup period
    start = 20  # For Donchian channel
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(atr[i]) or np.isnan(weekly_r1_aligned[i]) or np.isnan(weekly_s1_aligned[i]):
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
        volume_filter = volume[i] > vol_ma * 1.5  # Volume above 1.5x average
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price closes below Donchian lower OR below weekly S1
            # Stoploss: price drops 2*ATR below entry
            if (close[i] < lowest_low or
                close[i] < weekly_s1_aligned[i] or
                close[i] < entry_price - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            else:
                signals[i] = 0.25
            bars_since_exit += 1
        elif position == -1:  # short position
            # Exit: price closes above Donchian upper OR above weekly R1
            # Stoploss: price rises 2*ATR above entry
            if (close[i] > highest_high or
                close[i] > weekly_r1_aligned[i] or
                close[i] > entry_price + 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            else:
                signals[i] = -0.25
            bars_since_exit += 1
        else:
            # Look for entries: Donchian breakout + volume + weekly pivot direction
            # Minimum 10 bars between trades to prevent overtrading
            if bars_since_exit >= 10:
                bull_breakout = close[i] > highest_high
                bear_breakout = close[i] < lowest_low
                
                # Weekly pivot direction filter: only long above weekly pivot, short below
                above_pivot = close[i] > weekly_pivot_aligned[i]
                below_pivot = close[i] < weekly_pivot_aligned[i]
                
                if bull_breakout and volume_filter and above_pivot:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                    bars_since_exit = 0
                elif bear_breakout and volume_filter and below_pivot:
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

</think>

#!/usr/bin/env python3
"""
6h Donchian(20) Breakout + Weekly Pivot Direction + Volume Confirmation
Hypothesis: Weekly pivot levels provide structural support/resistance that works in both bull and bear markets.
Breakouts above weekly R1 or below weekly S1 with volume confirmation capture institutional interest.
Using 6h timeframe reduces trade frequency to avoid fee drag while capturing multi-day moves.
Weekly pivot filter ensures we only trade in direction of higher timeframe bias.
Target: 50-150 total trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_donchian20_weekly_pivot_vol_v1"
timeframe = "6h"
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
    
    # 14-period ATR for stoploss
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
    
    # === WEEKLY PIVOT DATA (MTF) - CALCULATE ONCE BEFORE LOOP ===
    df_weekly = get_htf_data(prices, '1w')  # Weekly data for pivot calculation
    if len(df_weekly) < 1:
        return np.zeros(n)
    
    # Calculate weekly pivot points: P = (H+L+C)/3
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    # Weekly pivot and support/resistance levels
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    weekly_r1 = 2 * weekly_pivot - weekly_low
    weekly_s1 = 2 * weekly_pivot - weekly_high
    weekly_r2 = weekly_pivot + (weekly_high - weekly_low)
    weekly_s2 = weekly_pivot - (weekly_high - weekly_low)
    weekly_r3 = weekly_high + 2 * (weekly_pivot - weekly_low)
    weekly_s3 = weekly_low - 2 * (weekly_high - weekly_pivot)
    
    # Align weekly levels to 6h timeframe (shifted by 1 week for no look-ahead)
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_weekly, weekly_pivot)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_weekly, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_weekly, weekly_s1)
    weekly_r2_aligned = align_htf_to_ltf(prices, df_weekly, weekly_r2)
    weekly_s2_aligned = align_htf_to_ltf(prices, df_weekly, weekly_s2)
    weekly_r3_aligned = align_htf_to_ltf(prices, df_weekly, weekly_r3)
    weekly_s3_aligned = align_htf_to_ltf(prices, df_weekly, weekly_s3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_exit = 0  # Bars since last exit to prevent overtrading
    
    # Start from warmup period
    start = 20  # For Donchian channel
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(atr[i]) or np.isnan(weekly_r1_aligned[i]) or np.isnan(weekly_s1_aligned[i]):
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
        volume_filter = volume[i] > vol_ma * 1.5  # Volume above 1.5x average
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price closes below Donchian lower OR below weekly S1
            # Stoploss: price drops 2*ATR below entry
            if (close[i] < lowest_low or
                close[i] < weekly_s1_aligned[i] or
                close[i] < entry_price - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            else:
                signals[i] = 0.25
            bars_since_exit += 1
        elif position == -1:  # short position
            # Exit: price closes above Donchian upper OR above weekly R1
            # Stoploss: price rises 2*ATR above entry
            if (close[i] > highest_high or
                close[i] > weekly_r1_aligned[i] or
                close[i] > entry_price + 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            else:
                signals[i] = -0.25
            bars_since_exit += 1
        else:
            # Look for entries: Donchian breakout + volume + weekly pivot direction
            # Minimum 10 bars between trades to prevent overtrading
            if bars_since_exit >= 10:
                bull_breakout = close[i] > highest_high
                bear_breakout = close[i] < lowest_low
                
                # Weekly pivot direction filter: only long above weekly pivot, short below
                above_pivot = close[i] > weekly_pivot_aligned[i]
                below_pivot = close[i] < weekly_pivot_aligned[i]
                
                if bull_breakout and volume_filter and above_pivot:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                    bars_since_exit = 0
                elif bear_breakout and volume_filter and below_pivot:
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