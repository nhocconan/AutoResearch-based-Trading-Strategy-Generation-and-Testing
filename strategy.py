#!/usr/bin/env python3
"""
6h Donchian(20) breakout with 1d weekly pivot direction and volume confirmation
Hypothesis: 6h Donchian breakouts capture intermediate-term momentum. Filter by 1d weekly pivot bias (price above/below weekly pivot) for trend alignment and volume confirmation for conviction. Works in bull (buy breakouts above weekly pivot) and bear (sell breakdowns below weekly pivot). Target: 75-200 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_donchian20_1d_pivot_vol_v1"
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
    
    # Get 1d data for weekly pivot calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate weekly pivot points (using prior week's OHLC)
    # Weekly pivot = (Prior Week High + Prior Week Low + Prior Week Close) / 3
    weekly_pivot = np.full(len(close_1d), np.nan)
    weekly_high = np.full(len(close_1d), np.nan)
    weekly_low = np.full(len(close_1d), np.nan)
    weekly_close = np.full(len(close_1d), np.nan)
    
    # We need to group by week - assuming data starts on Monday for simplicity
    # In practice, we'll use a rolling window approach for weekly values
    if len(close_1d) >= 5:
        for i in range(5, len(close_1d)):
            # Use prior 5 trading days (1 week) for pivot calculation
            weekly_high[i] = np.max(high_1d[i-5:i])
            weekly_low[i] = np.min(low_1d[i-5:i])
            weekly_close[i] = close_1d[i-1]  # Previous day's close
            if weekly_high[i] > 0 and weekly_low[i] > 0:
                weekly_pivot[i] = (weekly_high[i] + weekly_low[i] + weekly_close[i]) / 3.0
    
    # Weekly bias: above pivot = bullish, below = bearish
    weekly_bias = np.where(close_1d > weekly_pivot, 1, -1)
    
    # Align weekly bias to 6h timeframe
    weekly_bias_aligned = align_htf_to_ltf(prices, df_1d, weekly_bias)
    
    # Get 1d data for volume confirmation
    volume_1d = df_1d['volume'].values
    
    # 20-period average volume on 1d
    vol_ma_1d = np.full(len(volume_1d), np.nan)
    for i in range(20, len(volume_1d)):
        vol_ma_1d[i] = np.mean(volume_1d[i-20:i])
    
    # Align volume MA to 6h timeframe
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Donchian channels (20-period) from 6h data
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(20, n):
        upper[i] = np.max(high[i-20:i])
        lower[i] = np.min(low[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = 40  # Need enough data for Donchian and alignments
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(atr[i]) or np.isnan(weekly_bias_aligned[i]) or 
            np.isnan(upper[i]) or np.isnan(lower[i]) or
            np.isnan(vol_ma_1d_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            bars_since_entry += 1
            continue
        
        # Volume filter: current 6h volume > 1.5x 1d average volume (scaled)
        # Scale 1d volume to 6h: approx 1/4 of 1d volume (since 4x 6h in 1d)
        vol_threshold = vol_ma_1d_aligned[i] / 4.0 * 1.5
        volume_filter = volume[i] > vol_threshold
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price breaks below lower Donchian OR against weekly bias
            # Stoploss: price drops 2*ATR below entry
            if (close[i] < lower[i] or
                weekly_bias_aligned[i] == -1 or
                close[i] < entry_price - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.25
            bars_since_entry += 1
        elif position == -1:  # short position
            # Exit: price breaks above upper Donchian OR against weekly bias
            # Stoploss: price rises 2*ATR above entry
            if (close[i] > upper[i] or
                weekly_bias_aligned[i] == 1 or
                close[i] > entry_price + 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -0.25
            bars_since_entry += 1
        else:
            # Look for entries
            # Minimum holding period: only allow new entry after 6 bars flat
            if bars_since_entry >= 6:
                # Breakout entries: upper/lower with weekly bias
                bull_breakout = close[i] > upper[i]
                bear_breakout = close[i] < lower[i]
                
                # Long: breakout above upper with bullish weekly bias + volume
                if bull_breakout and weekly_bias_aligned[i] == 1 and volume_filter:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                # Short: breakdown below lower with bearish weekly bias + volume
                elif bear_breakout and weekly_bias_aligned[i] == -1 and volume_filter:
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
6h Donchian(20) breakout with 1d weekly pivot direction and volume confirmation
Hypothesis: 6h Donchian breakouts capture intermediate-term momentum. Filter by 1d weekly pivot bias (price above/below weekly pivot) for trend alignment and volume confirmation for conviction. Works in bull (buy breakouts above weekly pivot) and bear (sell breakdowns below weekly pivot). Target: 75-200 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_donchian20_1d_pivot_vol_v1"
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
    
    # Get 1d data for weekly pivot calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate weekly pivot points (using prior week's OHLC)
    # Weekly pivot = (Prior Week High + Prior Week Low + Prior Week Close) / 3
    weekly_pivot = np.full(len(close_1d), np.nan)
    weekly_high = np.full(len(close_1d), np.nan)
    weekly_low = np.full(len(close_1d), np.nan)
    weekly_close = np.full(len(close_1d), np.nan)
    
    # We need to group by week - assuming data starts on Monday for simplicity
    # In practice, we'll use a rolling window approach for weekly values
    if len(close_1d) >= 5:
        for i in range(5, len(close_1d)):
            # Use prior 5 trading days (1 week) for pivot calculation
            weekly_high[i] = np.max(high_1d[i-5:i])
            weekly_low[i] = np.min(low_1d[i-5:i])
            weekly_close[i] = close_1d[i-1]  # Previous day's close
            if weekly_high[i] > 0 and weekly_low[i] > 0:
                weekly_pivot[i] = (weekly_high[i] + weekly_low[i] + weekly_close[i]) / 3.0
    
    # Weekly bias: above pivot = bullish, below = bearish
    weekly_bias = np.where(close_1d > weekly_pivot, 1, -1)
    
    # Align weekly bias to 6h timeframe
    weekly_bias_aligned = align_htf_to_ltf(prices, df_1d, weekly_bias)
    
    # Get 1d data for volume confirmation
    volume_1d = df_1d['volume'].values
    
    # 20-period average volume on 1d
    vol_ma_1d = np.full(len(volume_1d), np.nan)
    for i in range(20, len(volume_1d)):
        vol_ma_1d[i] = np.mean(volume_1d[i-20:i])
    
    # Align volume MA to 6h timeframe
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Donchian channels (20-period) from 6h data
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(20, n):
        upper[i] = np.max(high[i-20:i])
        lower[i] = np.min(low[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = 40  # Need enough data for Donchian and alignments
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(atr[i]) or np.isnan(weekly_bias_aligned[i]) or 
            np.isnan(upper[i]) or np.isnan(lower[i]) or
            np.isnan(vol_ma_1d_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            bars_since_entry += 1
            continue
        
        # Volume filter: current 6h volume > 1.5x 1d average volume (scaled)
        # Scale 1d volume to 6h: approx 1/4 of 1d volume (since 4x 6h in 1d)
        vol_threshold = vol_ma_1d_aligned[i] / 4.0 * 1.5
        volume_filter = volume[i] > vol_threshold
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price breaks below lower Donchian OR against weekly bias
            # Stoploss: price drops 2*ATR below entry
            if (close[i] < lower[i] or
                weekly_bias_aligned[i] == -1 or
                close[i] < entry_price - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.25
            bars_since_entry += 1
        elif position == -1:  # short position
            # Exit: price breaks above upper Donchian OR against weekly bias
            # Stoploss: price rises 2*ATR above entry
            if (close[i] > upper[i] or
                weekly_bias_aligned[i] == 1 or
                close[i] > entry_price + 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -0.25
            bars_since_entry += 1
        else:
            # Look for entries
            # Minimum holding period: only allow new entry after 6 bars flat
            if bars_since_entry >= 6:
                # Breakout entries: upper/lower with weekly bias
                bull_breakout = close[i] > upper[i]
                bear_breakout = close[i] < lower[i]
                
                # Long: breakout above upper with bullish weekly bias + volume
                if bull_breakout and weekly_bias_aligned[i] == 1 and volume_filter:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                # Short: breakdown below lower with bearish weekly bias + volume
                elif bear_breakout and weekly_bias_aligned[i] == -1 and volume_filter:
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