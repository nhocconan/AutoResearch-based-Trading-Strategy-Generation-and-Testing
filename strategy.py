#!/usr/bin/env python3
"""
6h Donchian(20) Breakout + Weekly Pivot Direction + Volume Filter
Hypothesis: Combines price breakout momentum with weekly structural bias (from weekly pivots) and volume confirmation to filter false breakouts. Weekly pivot provides longer-term context to avoid counter-trend trades in ranging markets, improving performance in both bull and bear regimes. Designed for low trade frequency (target 50-150 total over 4 years) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_donchian20_weeklypivot_volume_v2"
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
        atr[0] = np.nan
        if len(tr) > 0:
            atr[1] = tr[0]
            for i in range(2, n):
                atr[i] = (tr[i-1] * 13 + atr[i-1]) / 14
    
    # Get weekly data (for pivot calculation) - ONCE before loop
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 1:
        # Not enough weekly data, fallback to no pivot filter
        weekly_pivot = np.full(n, np.nan)
        r4 = np.full(n, np.nan)
        s4 = np.full(n, np.nan)
    else:
        # Calculate weekly pivot points: P = (H+L+C)/3
        weekly_high = df_weekly['high'].values
        weekly_low = df_weekly['low'].values
        weekly_close = df_weekly['close'].values
        weekly_pivot_raw = (weekly_high + weekly_low + weekly_close) / 3.0
        
        # Weekly ranges for R4/S4: R4 = P + 3*(H-L), S4 = P - 3*(H-L)
        weekly_range = weekly_high - weekly_low
        r4_raw = weekly_pivot_raw + 3.0 * weekly_range
        s4_raw = weekly_pivot_raw - 3.0 * weekly_range
        
        # Align to 6h timeframe with shift(1) for completed bars only
        weekly_pivot = align_htf_to_ltf(prices, df_weekly, weekly_pivot_raw)
        r4 = align_htf_to_ltf(prices, df_weekly, r4_raw)
        s4 = align_htf_to_ltf(prices, df_weekly, s4_raw)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = 20  # For Donchian
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(atr[i]) or np.isnan(weekly_pivot[i]):
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
            # Look for entries: Donchian breakout + volume + weekly pivot filter
            bull_breakout = close[i] > highest_high
            bear_breakout = close[i] < lowest_low
            
            # Weekly pivot filter: only go long if price above weekly R4, short if below S4
            # This ensures we only trade in direction of stronger weekly momentum
            weekly_bull_bias = close[i] > r4[i]
            weekly_bear_bias = close[i] < s4[i]
            
            if bull_breakout and volume_filter and weekly_bull_bias:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif bear_breakout and volume_filter and weekly_bear_bias:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals

</think>
#!/usr/bin/env python3
"""
6h Donchian(20) Breakout + Weekly Pivot Direction + Volume Filter
Hypothesis: Combines price breakout momentum with weekly structural bias (from weekly pivots) and volume confirmation to filter false breakouts. Weekly pivot provides longer-term context to avoid counter-trend trades in ranging markets, improving performance in both bull and bear regimes. Designed for low trade frequency (target 50-150 total over 4 years) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_donchian20_weeklypivot_volume_v2"
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
        atr[0] = np.nan
        if len(tr) > 0:
            atr[1] = tr[0]
            for i in range(2, n):
                atr[i] = (tr[i-1] * 13 + atr[i-1]) / 14
    
    # Get weekly data (for pivot calculation) - ONCE before loop
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 1:
        # Not enough weekly data, fallback to no pivot filter
        weekly_pivot = np.full(n, np.nan)
        r4 = np.full(n, np.nan)
        s4 = np.full(n, np.nan)
    else:
        # Calculate weekly pivot points: P = (H+L+C)/3
        weekly_high = df_weekly['high'].values
        weekly_low = df_weekly['low'].values
        weekly_close = df_weekly['close'].values
        weekly_pivot_raw = (weekly_high + weekly_low + weekly_close) / 3.0
        
        # Weekly ranges for R4/S4: R4 = P + 3*(H-L), S4 = P - 3*(H-L)
        weekly_range = weekly_high - weekly_low
        r4_raw = weekly_pivot_raw + 3.0 * weekly_range
        s4_raw = weekly_pivot_raw - 3.0 * weekly_range
        
        # Align to 6h timeframe with shift(1) for completed bars only
        weekly_pivot = align_htf_to_ltf(prices, df_weekly, weekly_pivot_raw)
        r4 = align_htf_to_ltf(prices, df_weekly, r4_raw)
        s4 = align_htf_to_ltf(prices, df_weekly, s4_raw)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = 20  # For Donchian
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(atr[i]) or np.isnan(weekly_pivot[i]):
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
            # Look for entries: Donchian breakout + volume + weekly pivot filter
            bull_breakout = close[i] > highest_high
            bear_breakout = close[i] < lowest_low
            
            # Weekly pivot filter: only go long if price above weekly R4, short if below S4
            # This ensures we only trade in direction of stronger weekly momentum
            weekly_bull_bias = close[i] > r4[i]
            weekly_bear_bias = close[i] < s4[i]
            
            if bull_breakout and volume_filter and weekly_bull_bias:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif bear_breakout and volume_filter and weekly_bear_bias:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals