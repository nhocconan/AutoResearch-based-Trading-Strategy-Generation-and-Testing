#!/usr/bin/env python3
"""
6h Donchian(20) Breakout + Weekly Pivot Direction + Volume Spike
Hypothesis: Combines price breakout structure with weekly pivot bias to filter false breakouts.
In bull markets, weekly pivots provide support/resistance; in bear markets, they act as reversal zones.
Weekly pivot adds directional bias to reduce whipsaws. Volume spike confirms institutional interest.
Target: 80-150 total trades over 4 years (20-38/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_donchian20_weeklypivot_volume_v1"
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
    
    # Load weekly pivot data ONCE before loop
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) > 0:
        # Calculate weekly pivot points: (H + L + C) / 3
        weekly_pivot = (df_weekly['high'] + df_weekly['low'] + df_weekly['close']) / 3.0
        # Weekly support/resistance levels
        weekly_r1 = 2 * weekly_pivot - df_weekly['low']
        weekly_s1 = 2 * weekly_pivot - df_weekly['high']
        weekly_r2 = weekly_pivot + (df_weekly['high'] - df_weekly['low'])
        weekly_s2 = weekly_pivot - (df_weekly['high'] - df_weekly['low'])
        # Align to 6h timeframe
        pivot_aligned = align_htf_to_ltf(prices, df_weekly, weekly_pivot.values)
        r1_aligned = align_htf_to_ltf(prices, df_weekly, weekly_r1.values)
        s1_aligned = align_htf_to_ltf(prices, df_weekly, weekly_s1.values)
        r2_aligned = align_htf_to_ltf(prices, df_weekly, weekly_r2.values)
        s2_aligned = align_htf_to_ltf(prices, df_weekly, weekly_s2.values)
    else:
        # Fallback if no weekly data
        pivot_aligned = np.full(n, np.nan)
        r1_aligned = np.full(n, np.nan)
        s1_aligned = np.full(n, np.nan)
        r2_aligned = np.full(n, np.nan)
        s2_aligned = np.full(n, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = 20  # For Donchian
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(atr[i]) or np.isnan(pivot_aligned[i]):
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
            # Exit: price closes below Donchian lower OR breaks weekly S1
            # Stoploss: price drops 2*ATR below entry
            if (close[i] < lowest_low or
                close[i] < s1_aligned[i] or
                close[i] < entry_price - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.25
            bars_since_entry += 1
        elif position == -1:  # short position
            # Exit: price closes above Donchian upper OR breaks weekly R1
            # Stoploss: price rises 2*ATR above entry
            if (close[i] > highest_high or
                close[i] > r1_aligned[i] or
                close[i] > entry_price + 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -0.25
            bars_since_entry += 1
        else:
            # Look for entries: Donchian breakout + volume + weekly pivot filter
            # Minimum holding period: only allow new entry after 20 bars flat
            if bars_since_entry >= 20:
                bull_breakout = close[i] > highest_high
                bear_breakout = close[i] < lowest_low
                
                # Weekly pivot filter: long above pivot, short below pivot
                pivot_bias_long = close[i] > pivot_aligned[i]
                pivot_bias_short = close[i] < pivot_aligned[i]
                
                if bull_breakout and volume_filter and pivot_bias_long:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                elif bear_breakout and volume_filter and pivot_bias_short:
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

</think>

#!/usr/bin/env python3
"""
6h Donchian(20) Breakout + Weekly Pivot Direction + Volume Spike
Hypothesis: Combines price breakout structure with weekly pivot bias to filter false breakouts.
In bull markets, weekly pivots provide support/resistance; in bear markets, they act as reversal zones.
Weekly pivot adds directional bias to reduce whipsaws. Volume spike confirms institutional interest.
Target: 80-150 total trades over 4 years (20-38/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_donchian20_weeklypivot_volume_v1"
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
    
    # Load weekly pivot data ONCE before loop
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) > 0:
        # Calculate weekly pivot points: (H + L + C) / 3
        weekly_pivot = (df_weekly['high'] + df_weekly['low'] + df_weekly['close']) / 3.0
        # Weekly support/resistance levels
        weekly_r1 = 2 * weekly_pivot - df_weekly['low']
        weekly_s1 = 2 * weekly_pivot - df_weekly['high']
        weekly_r2 = weekly_pivot + (df_weekly['high'] - df_weekly['low'])
        weekly_s2 = weekly_pivot - (df_weekly['high'] - df_weekly['low'])
        # Align to 6h timeframe
        pivot_aligned = align_htf_to_ltf(prices, df_weekly, weekly_pivot.values)
        r1_aligned = align_htf_to_ltf(prices, df_weekly, weekly_r1.values)
        s1_aligned = align_htf_to_ltf(prices, df_weekly, weekly_s1.values)
        r2_aligned = align_htf_to_ltf(prices, df_weekly, weekly_r2.values)
        s2_aligned = align_htf_to_ltf(prices, df_weekly, weekly_s2.values)
    else:
        # Fallback if no weekly data
        pivot_aligned = np.full(n, np.nan)
        r1_aligned = np.full(n, np.nan)
        s1_aligned = np.full(n, np.nan)
        r2_aligned = np.full(n, np.nan)
        s2_aligned = np.full(n, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = 20  # For Donchian
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(atr[i]) or np.isnan(pivot_aligned[i]):
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
            # Exit: price closes below Donchian lower OR breaks weekly S1
            # Stoploss: price drops 2*ATR below entry
            if (close[i] < lowest_low or
                close[i] < s1_aligned[i] or
                close[i] < entry_price - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.25
            bars_since_entry += 1
        elif position == -1:  # short position
            # Exit: price closes above Donchian upper OR breaks weekly R1
            # Stoploss: price rises 2*ATR above entry
            if (close[i] > highest_high or
                close[i] > r1_aligned[i] or
                close[i] > entry_price + 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -0.25
            bars_since_entry += 1
        else:
            # Look for entries: Donchian breakout + volume + weekly pivot filter
            # Minimum holding period: only allow new entry after 20 bars flat
            if bars_since_entry >= 20:
                bull_breakout = close[i] > highest_high
                bear_breakout = close[i] < lowest_low
                
                # Weekly pivot filter: long above pivot, short below pivot
                pivot_bias_long = close[i] > pivot_aligned[i]
                pivot_bias_short = close[i] < pivot_aligned[i]
                
                if bull_breakout and volume_filter and pivot_bias_long:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                elif bear_breakout and volume_filter and pivot_bias_short:
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