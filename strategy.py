#!/usr/bin/env python3
"""
6h Donchian(20) Breakout + Volume Spike + Weekly Pivot Direction + ATR Stoploss
Hypothesis: Combines price breakouts with volume confirmation and weekly pivot direction filter.
Weekly pivot (from 1d data) provides institutional reference points: price tends to respect these levels.
In bull markets, breaks above weekly R3/R4 with volume continue upward; in bear markets, breaks below S3/S4 with volume continue downward.
Volume filter ensures only high-momentum breaks are taken. Target: 100-200 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_donchian20_weekly_pivot_vol_v2"
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
    
    # Get weekly pivot levels from 1d data (use 1d as proxy for weekly)
    # Calculate pivots for previous week using 1d OHLC data
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:  # Need at least a week of data
        return np.zeros(n)
    
    # Calculate weekly pivot points using prior week's data
    # For each point, we use the prior week's high, low, close
    weekly_high = df_1d['high'].rolling(window=5, min_periods=5).max().shift(1)  # Prior week high
    weekly_low = df_1d['low'].rolling(window=5, min_periods=5).min().shift(1)    # Prior week low
    weekly_close = df_1d['close'].rolling(window=5, min_periods=5).last().shift(1)  # Prior week close
    
    # Standard pivot point calculation
    pivot = (weekly_high + weekly_low + weekly_close) / 3
    r1 = 2 * pivot - weekly_low
    s1 = 2 * pivot - weekly_high
    r2 = pivot + (weekly_high - weekly_low)
    s2 = pivot - (weekly_high - weekly_low)
    r3 = weekly_high + 2 * (pivot - weekly_low)
    s3 = weekly_low - 2 * (weekly_high - pivot)
    r4 = weekly_high + 3 * (pivot - weekly_low)
    s4 = weekly_low - 3 * (weekly_high - pivot)
    
    # Align pivot levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot.values)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1.values)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1.values)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2.values)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2.values)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3.values)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3.values)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4.values)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = 20  # For Donchian
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(atr[i]):
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
        volume_filter = volume[i] > vol_ma * 1.5  # Volume > 1.5x average
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price closes below Donchian lower OR breaks below S3 (strong bearish signal)
            # Stoploss: price drops 2*ATR below entry
            if (close[i] < lowest_low or
                close[i] < s3_aligned[i] or
                close[i] < entry_price - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.25
            bars_since_entry += 1
        elif position == -1:  # short position
            # Exit: price closes above Donchian upper OR breaks above R3 (strong bullish signal)
            # Stoploss: price rises 2*ATR above entry
            if (close[i] > highest_high or
                close[i] > r3_aligned[i] or
                close[i] > entry_price + 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -0.25
            bars_since_entry += 1
        else:
            # Look for entries: Donchian breakout + volume filter
            # Minimum holding period: only allow new entry after 15 bars flat
            if bars_since_entry >= 15:
                bull_breakout = close[i] > highest_high
                bear_breakout = close[i] < lowest_low
                
                # Additional filter: breakout should be in direction of weekly pivot bias
                # For longs: price should be above weekly pivot (bullish bias)
                # For shorts: price should be below weekly pivot (bearish bias)
                bullish_bias = close[i] > pivot_aligned[i]
                bearish_bias = close[i] < pivot_aligned[i]
                
                if bull_breakout and volume_filter and bullish_bias:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                elif bear_breakout and volume_filter and bearish_bias:
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
6h Donchian(20) Breakout + Volume Spike + Weekly Pivot Direction + ATR Stoploss
Hypothesis: Combines price breakouts with volume confirmation and weekly pivot direction filter.
Weekly pivot (from 1d data) provides institutional reference points: price tends to respect these levels.
In bull markets, breaks above weekly R3/R4 with volume continue upward; in bear markets, breaks below S3/S4 with volume continue downward.
Volume filter ensures only high-momentum breaks are taken. Target: 100-200 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_donchian20_weekly_pivot_vol_v2"
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
    
    # Get weekly pivot levels from 1d data (use 1d as proxy for weekly)
    # Calculate pivots for previous week using 1d OHLC data
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:  # Need at least a week of data
        return np.zeros(n)
    
    # Calculate weekly pivot points using prior week's data
    # For each point, we use the prior week's high, low, close
    weekly_high = df_1d['high'].rolling(window=5, min_periods=5).max().shift(1)  # Prior week high
    weekly_low = df_1d['low'].rolling(window=5, min_periods=5).min().shift(1)    # Prior week low
    weekly_close = df_1d['close'].rolling(window=5, min_periods=5).last().shift(1)  # Prior week close
    
    # Standard pivot point calculation
    pivot = (weekly_high + weekly_low + weekly_close) / 3
    r1 = 2 * pivot - weekly_low
    s1 = 2 * pivot - weekly_high
    r2 = pivot + (weekly_high - weekly_low)
    s2 = pivot - (weekly_high - weekly_low)
    r3 = weekly_high + 2 * (pivot - weekly_low)
    s3 = weekly_low - 2 * (weekly_high - pivot)
    r4 = weekly_high + 3 * (pivot - weekly_low)
    s4 = weekly_low - 3 * (weekly_high - pivot)
    
    # Align pivot levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot.values)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1.values)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1.values)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2.values)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2.values)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3.values)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3.values)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4.values)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = 20  # For Donchian
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(atr[i]):
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
        volume_filter = volume[i] > vol_ma * 1.5  # Volume > 1.5x average
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price closes below Donchian lower OR breaks below S3 (strong bearish signal)
            # Stoploss: price drops 2*ATR below entry
            if (close[i] < lowest_low or
                close[i] < s3_aligned[i] or
                close[i] < entry_price - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.25
            bars_since_entry += 1
        elif position == -1:  # short position
            # Exit: price closes above Donchian upper OR breaks above R3 (strong bullish signal)
            # Stoploss: price rises 2*ATR above entry
            if (close[i] > highest_high or
                close[i] > r3_aligned[i] or
                close[i] > entry_price + 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -0.25
            bars_since_entry += 1
        else:
            # Look for entries: Donchian breakout + volume filter
            # Minimum holding period: only allow new entry after 15 bars flat
            if bars_since_entry >= 15:
                bull_breakout = close[i] > highest_high
                bear_breakout = close[i] < lowest_low
                
                # Additional filter: breakout should be in direction of weekly pivot bias
                # For longs: price should be above weekly pivot (bullish bias)
                # For shorts: price should be below weekly pivot (bearish bias)
                bullish_bias = close[i] > pivot_aligned[i]
                bearish_bias = close[i] < pivot_aligned[i]
                
                if bull_breakout and volume_filter and bullish_bias:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                elif bear_breakout and volume_filter and bearish_bias:
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