#!/usr/bin/env python3
"""
4h Donchian(20) breakout with volume and trend filter
Hypothesis: Breakouts from 20-period Donchian channels on 4h timeframe capture medium-term momentum.
Filtered by volume > 1.5x average volume and trend alignment (price > SMA50 on 4h).
Works in bull markets (buy breakouts) and bear markets (sell breakdowns).
Target: 80-150 total trades over 4 years (20-38/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian20_volume_trend_v1"
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
    
    # SMA50 on 4h for trend filter
    sma50 = np.full(n, np.nan)
    if n >= 50:
        for i in range(50, n):
            sma50[i] = np.mean(close[i-50:i])
    
    # Average volume for volume filter (20-period)
    vol_ma = np.full(n, np.nan)
    if n >= 20:
        for i in range(20, n):
            vol_ma[i] = np.mean(volume[i-20:i])
    
    # Donchian channels (20-period)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(20, n):
        upper[i] = np.max(high[i-20:i])
        lower[i] = np.min(low[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_exit = 0  # bars since last exit to prevent overtrading
    
    # Start from warmup period
    start = 50  # Need SMA50 and Donchian
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(atr[i]) or np.isnan(sma50[i]) or np.isnan(vol_ma[i]) or \
           np.isnan(upper[i]) or np.isnan(lower[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            bars_since_exit += 1
            continue
        
        # Volume filter: current volume > 1.5x average volume
        volume_filter = volume[i] > vol_ma[i] * 1.5
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price breaks below lower Donchian OR stoploss hit
            if close[i] < lower[i] or close[i] < entry_price - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            else:
                signals[i] = 0.25
            bars_since_exit += 1
        elif position == -1:  # short position
            # Exit: price breaks above upper Donchian OR stoploss hit
            if close[i] > upper[i] or close[i] > entry_price + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            else:
                signals[i] = -0.25
            bars_since_exit += 1
        else:
            # Look for entries with minimum bars since exit
            if bars_since_exit >= 8:  # Minimum 8 bars between trades
                # Breakout entries
                bull_breakout = close[i] > upper[i]
                bear_breakout = close[i] < lower[i]
                
                # Trend filter: price > SMA50 for long, price < SMA50 for short
                # Note: In strong trends, price stays above/below SMA
                if bull_breakout and close[i] > sma50[i] and volume_filter:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                    bars_since_exit = 0
                elif bear_breakout and close[i] < sma50[i] and volume_filter:
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
4h Donchian(20) breakout with volume and trend filter
Hypothesis: Breakouts from 20-period Donchian channels on 4h timeframe capture medium-term momentum.
Filtered by volume > 1.5x average volume and trend alignment (price > SMA50 on 4h).
Works in bull markets (buy breakouts) and bear markets (sell breakdowns).
Target: 80-150 total trades over 4 years (20-38/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian20_volume_trend_v1"
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
    
    # SMA50 on 4h for trend filter
    sma50 = np.full(n, np.nan)
    if n >= 50:
        for i in range(50, n):
            sma50[i] = np.mean(close[i-50:i])
    
    # Average volume for volume filter (20-period)
    vol_ma = np.full(n, np.nan)
    if n >= 20:
        for i in range(20, n):
            vol_ma[i] = np.mean(volume[i-20:i])
    
    # Donchian channels (20-period)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(20, n):
        upper[i] = np.max(high[i-20:i])
        lower[i] = np.min(low[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_exit = 0  # bars since last exit to prevent overtrading
    
    # Start from warmup period
    start = 50  # Need SMA50 and Donchian
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(atr[i]) or np.isnan(sma50[i]) or np.isnan(vol_ma[i]) or \
           np.isnan(upper[i]) or np.isnan(lower[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            bars_since_exit += 1
            continue
        
        # Volume filter: current volume > 1.5x average volume
        volume_filter = volume[i] > vol_ma[i] * 1.5
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price breaks below lower Donchian OR stoploss hit
            if close[i] < lower[i] or close[i] < entry_price - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            else:
                signals[i] = 0.25
            bars_since_exit += 1
        elif position == -1:  # short position
            # Exit: price breaks above upper Donchian OR stoploss hit
            if close[i] > upper[i] or close[i] > entry_price + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            else:
                signals[i] = -0.25
            bars_since_exit += 1
        else:
            # Look for entries with minimum bars since exit
            if bars_since_exit >= 8:  # Minimum 8 bars between trades
                # Breakout entries
                bull_breakout = close[i] > upper[i]
                bear_breakout = close[i] < lower[i]
                
                # Trend filter: price > SMA50 for long, price < SMA50 for short
                # Note: In strong trends, price stays above/below SMA
                if bull_breakout and close[i] > sma50[i] and volume_filter:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                    bars_since_exit = 0
                elif bear_breakout and close[i] < sma50[i] and volume_filter:
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