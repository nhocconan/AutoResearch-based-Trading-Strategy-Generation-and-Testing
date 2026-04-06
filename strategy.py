#!/usr/bin/env python3
"""
6h Donchian(20) Breakout + Weekly Trend Filter + Volume Confirmation
Hypothesis: Donchian breakouts on 6h capture medium-term momentum; weekly trend filter avoids counter-trend trades; volume confirms institutional participation. Designed for 50-150 total trades over 4 years (12-37/year) with proper risk control.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_donchian20_weekly_trend_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data for trend filter (once before loop)
    df_weekly = get_htf_data(prices, '1w')
    
    # Weekly EMA(21) for trend direction
    close_weekly = df_weekly['close'].values
    ema_weekly = pd.Series(close_weekly).ewm(span=21, adjust=False).mean().values
    
    # Align weekly trend to 6h timeframe (shifted by 1 weekly bar for completed bars only)
    trend_weekly = align_htf_to_ltf(prices, df_weekly, ema_weekly)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from warmup period
    start = 20  # For Donchian channel
    
    for i in range(start, n):
        # Skip if weekly trend not available
        if np.isnan(trend_weekly[i]):
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
        
        # Check exits
        if position == 1:  # long position
            # Exit: price closes below Donchian lower OR weekly trend turns bearish
            if close[i] < lowest_low or close[i] < trend_weekly[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price closes above Donchian upper OR weekly trend turns bullish
            if close[i] > highest_high or close[i] > trend_weekly[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout + volume + weekly trend alignment
            bull_breakout = close[i] > highest_high
            bear_breakout = close[i] < lowest_low
            
            if i >= 20 and bull_breakout and volume_filter and close[i] > trend_weekly[i]:
                signals[i] = 0.25
                position = 1
            elif i >= 20 and bear_breakout and volume_filter and close[i] < trend_weekly[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals

</think>
#!/usr/bin/env python3
"""
6h Donchian(20) Breakout + Weekly Trend Filter + Volume Confirmation
Hypothesis: Donchian breakouts on 6h capture medium-term momentum; weekly trend filter avoids counter-trend trades; volume confirms institutional participation. Designed for 50-150 total trades over 4 years (12-37/year) with proper risk control.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_donchian20_weekly_trend_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data for trend filter (once before loop)
    df_weekly = get_htf_data(prices, '1w')
    
    # Weekly EMA(21) for trend direction
    close_weekly = df_weekly['close'].values
    ema_weekly = pd.Series(close_weekly).ewm(span=21, adjust=False).mean().values
    
    # Align weekly trend to 6h timeframe (shifted by 1 weekly bar for completed bars only)
    trend_weekly = align_htf_to_ltf(prices, df_weekly, ema_weekly)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from warmup period
    start = 20  # For Donchian channel
    
    for i in range(start, n):
        # Skip if weekly trend not available
        if np.isnan(trend_weekly[i]):
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
        
        # Check exits
        if position == 1:  # long position
            # Exit: price closes below Donchian lower OR weekly trend turns bearish
            if close[i] < lowest_low or close[i] < trend_weekly[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price closes above Donchian upper OR weekly trend turns bullish
            if close[i] > highest_high or close[i] > trend_weekly[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout + volume + weekly trend alignment
            bull_breakout = close[i] > highest_high
            bear_breakout = close[i] < lowest_low
            
            if i >= 20 and bull_breakout and volume_filter and close[i] > trend_weekly[i]:
                signals[i] = 0.25
                position = 1
            elif i >= 20 and bear_breakout and volume_filter and close[i] < trend_weekly[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals