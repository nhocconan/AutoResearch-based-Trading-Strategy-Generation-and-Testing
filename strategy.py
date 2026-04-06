#!/usr/bin/env python3
"""
6h Donchian(20) Breakout + 1D Daily Range Position + Volume Filter
Hypothesis: Breakouts in the direction of the prior day's range (close relative to mid-point)
capture institutional flow, volume confirms, and reduces false signals in ranging markets.
Designed for 60-120 total trades over 4 years (15-30/year) to minimize fee drag.
Works in bull (breakouts with trend) and bear (mean reversion fails, breakouts persist).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_donchian20_1d_rangepos_vol"
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
    
    # 14-period ATR for stoploss
    atr = np.full(n, np.nan)
    if n >= 14:
        tr = np.maximum(
            high[1:] - low[1:],
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        )
        if len(tr) > 0:
            atr[0] = np.nan
            atr[1] = tr[0] if n > 1 else np.nan
            for i in range(2, n):
                atr[i] = (tr[i-1] * 13 + atr[i-1]) / 14
    
    # Load 1D data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) == 0:
        return np.zeros(n)
    
    # Calculate 1D range position: (close - low) / (high - low)
    # 0 = at low, 0.5 = mid, 1 = at high
    daily_range = df_1d['high'] - df_1d['low']
    # Avoid division by zero
    daily_range = np.where(daily_range == 0, 1e-10, daily_range)
    range_pos = (df_1d['close'] - df_1d['low']) / daily_range
    
    # Align to 6h timeframe (shifted by 1 for prior day only)
    range_pos_aligned = align_htf_to_ltf(prices, df_1d, range_pos.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = 20  # For Donchian
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(atr[i]) or np.isnan(range_pos_aligned[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Donchian channel (20-period)
        highest_high = np.max(high[i-20:i])
        lowest_low = np.min(low[i-20:i])
        
        # Volume filter (20-period average)
        vol_ma = np.mean(volume[i-20:i])
        volume_filter = volume[i] > vol_ma * 1.5
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price closes below Donchian lower
            # Stoploss: price drops 2*ATR below entry
            if (close[i] < lowest_low or
                close[i] < entry_price - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price closes above Donchian upper
            # Stoploss: price rises 2*ATR above entry
            if (close[i] > highest_high or
                close[i] > entry_price + 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout + volume + 1D range position filter
            bull_breakout = close[i] > highest_high
            bear_breakout = close[i] < lowest_low
            
            # Range position filter:
            # Long when prior day closed in upper half (bullish bias)
            # Short when prior day closed in lower half (bearish bias)
            range_pos_val = range_pos_aligned[i]
            long_bias = range_pos_val > 0.5  # Prior day closed above midpoint
            short_bias = range_pos_val < 0.5  # Prior day closed below midpoint
            
            if bull_breakout and volume_filter and long_bias:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif bear_breakout and volume_filter and short_bias:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals

</think>
#!/usr/bin/env python3
"""
6h Donchian(20) Breakout + 1D Daily Range Position + Volume Filter
Hypothesis: Breakouts in the direction of the prior day's range (close relative to mid-point)
capture institutional flow, volume confirms, and reduces false signals in ranging markets.
Designed for 60-120 total trades over 4 years (15-30/year) to minimize fee drag.
Works in bull (breakouts with trend) and bear (mean reversion fails, breakouts persist).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_donchian20_1d_rangepos_vol"
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
    
    # 14-period ATR for stoploss
    atr = np.full(n, np.nan)
    if n >= 14:
        tr = np.maximum(
            high[1:] - low[1:],
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        )
        if len(tr) > 0:
            atr[0] = np.nan
            atr[1] = tr[0] if n > 1 else np.nan
            for i in range(2, n):
                atr[i] = (tr[i-1] * 13 + atr[i-1]) / 14
    
    # Load 1D data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) == 0:
        return np.zeros(n)
    
    # Calculate 1D range position: (close - low) / (high - low)
    # 0 = at low, 0.5 = mid, 1 = at high
    daily_range = df_1d['high'] - df_1d['low']
    # Avoid division by zero
    daily_range = np.where(daily_range == 0, 1e-10, daily_range)
    range_pos = (df_1d['close'] - df_1d['low']) / daily_range
    
    # Align to 6h timeframe (shifted by 1 for prior day only)
    range_pos_aligned = align_htf_to_ltf(prices, df_1d, range_pos.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = 20  # For Donchian
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(atr[i]) or np.isnan(range_pos_aligned[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Donchian channel (20-period)
        highest_high = np.max(high[i-20:i])
        lowest_low = np.min(low[i-20:i])
        
        # Volume filter (20-period average)
        vol_ma = np.mean(volume[i-20:i])
        volume_filter = volume[i] > vol_ma * 1.5
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price closes below Donchian lower
            # Stoploss: price drops 2*ATR below entry
            if (close[i] < lowest_low or
                close[i] < entry_price - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price closes above Donchian upper
            # Stoploss: price rises 2*ATR above entry
            if (close[i] > highest_high or
                close[i] > entry_price + 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout + volume + 1D range position filter
            bull_breakout = close[i] > highest_high
            bear_breakout = close[i] < lowest_low
            
            # Range position filter:
            # Long when prior day closed in upper half (bullish bias)
            # Short when prior day closed in lower half (bearish bias)
            range_pos_val = range_pos_aligned[i]
            long_bias = range_pos_val > 0.5  # Prior day closed above midpoint
            short_bias = range_pos_val < 0.5  # Prior day closed below midpoint
            
            if bull_breakout and volume_filter and long_bias:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif bear_breakout and volume_filter and short_bias:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals

</think>