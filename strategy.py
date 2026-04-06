#!/usr/bin/env python3
"""
6h Donchian Breakout with Weekly Pivot Direction and Volume Confirmation
Hypothesis: 6h Donchian(20) breakouts aligned with weekly pivot direction (above/below weekly pivot)
and volume confirmation capture institutional moves in both bull and bear markets.
Weekly pivot provides directional bias from higher timeframe, reducing false breakouts.
Volume ensures participation. Target: 50-150 total trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_donchian_breakout_weekly_pivot_volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Load 6h and weekly data (once before loop)
    df_6h = get_htf_data(prices, '6h')
    df_weekly = get_htf_data(prices, '1w')
    
    # 6h Donchian channels (20-period)
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    high_roll = pd.Series(high_6h).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low_6h).rolling(window=20, min_periods=20).min().values
    
    # Weekly pivot point (standard calculation: (H+L+C)/3)
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    close_weekly = df_weekly['close'].values
    weekly_pivot = (high_weekly + low_weekly + close_weekly) / 3.0
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_weekly, weekly_pivot)
    
    # 6h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 6h volume filter (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # 6h ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = 20  # For Donchian(20)
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits
        if position == 1:  # long position
            # Exit: price breaks below lower Donchian OR stoploss
            if (close[i] <= low_roll[i] or
                close[i] <= entry_price - 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price breaks above upper Donchian OR stoploss
            if (close[i] >= high_roll[i] or
                close[i] >= entry_price + 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout + weekly pivot direction + volume
            long_breakout = close[i] > high_roll[i-1]  # Break above previous period's high
            short_breakout = close[i] < low_roll[i-1]  # Break below previous period's low
            
            # Weekly pivot direction: above pivot = bullish bias, below = bearish bias
            above_weekly_pivot = close[i] > weekly_pivot_aligned[i]
            below_weekly_pivot = close[i] < weekly_pivot_aligned[i]
            
            # Volume confirmation: current volume > 1.5x average
            vol_confirm = volume[i] > (1.5 * vol_ma[i])
            
            if long_breakout and above_weekly_pivot and vol_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif short_breakout and below_weekly_pivot and vol_confirm:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals

</think>
#!/usr/bin/env python3
"""
6h Donchian Breakout with Weekly Pivot Direction and Volume Confirmation
Hypothesis: 6h Donchian(20) breakouts aligned with weekly pivot direction (above/below weekly pivot)
and volume confirmation capture institutional moves in both bull and bear markets.
Weekly pivot provides directional bias from higher timeframe, reducing false breakouts.
Volume ensures participation. Target: 50-150 total trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_donchian_breakout_weekly_pivot_volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Load 6h and weekly data (once before loop)
    df_6h = get_htf_data(prices, '6h')
    df_weekly = get_htf_data(prices, '1w')
    
    # 6h Donchian channels (20-period)
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    high_roll = pd.Series(high_6h).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low_6h).rolling(window=20, min_periods=20).min().values
    
    # Weekly pivot point (standard calculation: (H+L+C)/3)
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    close_weekly = df_weekly['close'].values
    weekly_pivot = (high_weekly + low_weekly + close_weekly) / 3.0
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_weekly, weekly_pivot)
    
    # 6h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 6h volume filter (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # 6h ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = 20  # For Donchian(20)
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits
        if position == 1:  # long position
            # Exit: price breaks below lower Donchian OR stoploss
            if (close[i] <= low_roll[i] or
                close[i] <= entry_price - 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price breaks above upper Donchian OR stoploss
            if (close[i] >= high_roll[i] or
                close[i] >= entry_price + 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout + weekly pivot direction + volume
            long_breakout = close[i] > high_roll[i-1]  # Break above previous period's high
            short_breakout = close[i] < low_roll[i-1]  # Break below previous period's low
            
            # Weekly pivot direction: above pivot = bullish bias, below = bearish bias
            above_weekly_pivot = close[i] > weekly_pivot_aligned[i]
            below_weekly_pivot = close[i] < weekly_pivot_aligned[i]
            
            # Volume confirmation: current volume > 1.5x average
            vol_confirm = volume[i] > (1.5 * vol_ma[i])
            
            if long_breakout and above_weekly_pivot and vol_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif short_breakout and below_weekly_pivot and vol_confirm:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals

</think>
#!/usr/bin/env python3
"""
6h Donchian Breakout with Weekly Pivot Direction and Volume Confirmation
Hypothesis: 6h Donchian(20) breakouts aligned with weekly pivot direction (above/below weekly pivot)
and volume confirmation capture institutional moves in both bull and bear markets.
Weekly pivot provides directional bias from higher timeframe, reducing false breakouts.
Volume ensures participation. Target: 50-150 total trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_donchian_breakout_weekly_pivot_volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Load 6h and weekly data (once before loop)
    df_6h = get_htf_data(prices, '6h')
    df_weekly = get_htf_data(prices, '1w')
    
    # 6h Donchian channels (20-period)
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    high_roll = pd.Series(high_6h).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low_6h).rolling(window=20, min_periods=20).min().values
    
    # Weekly pivot point (standard calculation: (H+L+C)/3)
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    close_weekly = df_weekly['close'].values
    weekly_pivot = (high_weekly + low_weekly + close_weekly) / 3.0
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_weekly, weekly_pivot)
    
    # 6h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 6h volume filter (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # 6h ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = 20  # For Donchian(20)
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits
        if position == 1:  # long position
            # Exit: price breaks below lower Donchian OR stoploss
            if (close[i] <= low_roll[i] or
                close[i] <= entry_price - 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price breaks above upper Donchian OR stoploss
            if (close[i] >= high_roll[i] or
                close[i] >= entry_price + 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout + weekly pivot direction + volume
            long_breakout = close[i] > high_roll[i-1]  # Break above previous period's high
            short_breakout = close[i] < low_roll[i-1]  # Break below previous period's low
            
            # Weekly pivot direction: above pivot = bullish bias, below = bearish bias
            above_weekly_pivot = close[i] > weekly_pivot_aligned[i]
            below_weekly_pivot = close[i] < weekly_pivot_aligned[i]
            
            # Volume confirmation: current volume > 1.5x average
            vol_confirm = volume[i] > (1.5 * vol_ma[i])
            
            if long_breakout and above_weekly_pivot and vol_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif short_breakout and below_weekly_pivot and vol_confirm:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals

</think>
#!/usr/bin/env python3
"""
6h Donchian Breakout with Weekly Pivot Direction and Volume Confirmation
Hypothesis: 6h Donchian(20) breakouts aligned with weekly pivot direction (above/below weekly pivot)
and volume confirmation capture institutional moves in both bull and bear markets.
Weekly pivot provides directional bias from higher timeframe, reducing false breakouts.
Volume ensures participation. Target: 50-150 total trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_donchian_breakout_weekly_pivot_volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Load 6h and weekly data (once before loop)
    df_6h = get_htf_data(prices, '6h')
    df_weekly = get_htf_data(prices, '1w')
    
    # 6h Donchian channels (20-period)
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    high_roll = pd.Series(high_6h).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low_6h).rolling(window=20, min_periods=20).min().values
    
    # Weekly pivot point (standard calculation: (H+L+C)/3)
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    close_weekly = df_weekly['close'].values
    weekly_pivot = (high_weekly + low_weekly + close_weekly) / 3.0
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_weekly, weekly_pivot)
    
    # 6h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 6h volume filter (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # 6h ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = 20  # For Donchian(20)
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits
        if position == 1:  # long position
            # Exit: price breaks below lower Donchian OR stoploss
            if (close[i] <= low_roll[i] or
                close[i] <= entry_price - 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price breaks above upper Donchian OR stoploss
            if (close[i] >= high_roll[i] or
                close[i] >= entry_price + 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout + weekly pivot direction + volume
            long_breakout = close[i] > high_roll[i-1]  # Break above previous period's high
            short_breakout = close[i] < low_roll[i-1]  # Break below previous period's low
            
            # Weekly pivot direction: above pivot = bullish bias, below = bearish bias
            above_weekly_pivot = close[i] > weekly_pivot_aligned[i]
            below_weekly_pivot = close[i] < weekly_pivot_aligned[i]
            
            # Volume confirmation: current volume > 1.5x average
            vol_confirm = volume[i] > (1.5 * vol_ma[i])
            
            if long_breakout and above_weekly_pivot and vol_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif short_breakout and below_weekly_pivot and vol_confirm:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals

</think>
#!/usr/bin/env python3
"""
6h Donchian Breakout with Weekly Pivot Direction and Volume Confirmation
Hypothesis: 6h Donchian(20) breakouts aligned with weekly pivot direction (above/below weekly pivot)
and volume confirmation capture institutional moves in both bull and bear markets.
Weekly pivot provides directional bias from higher timeframe, reducing false breakouts.
Volume ensures participation. Target: 50-150 total trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_donchian_breakout_weekly_pivot_volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Load 6h and weekly data (once before loop)
    df_6h = get_htf_data(prices, '6h')
    df_weekly = get_htf_data(prices, '1w')
    
    # 6h Donchian channels (20-period)
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    high_roll = pd.Series(high_6h).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low_6h).rolling(window=20, min_periods=20).min().values
    
    # Weekly pivot point (standard calculation: (H+L+C)/3)
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    close_weekly = df_weekly['close'].values
    weekly_pivot = (high_weekly + low_weekly + close_weekly) / 3.0
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_weekly, weekly_pivot)
    
    # 6h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 6h volume filter (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # 6h ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = 20  # For Donchian(20)
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits
        if position == 1:  # long position
            # Exit: price breaks below lower Donchian OR stoploss
            if (close[i] <= low_roll[i] or
                close[i] <= entry_price - 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price breaks above upper Donchian OR stoploss
            if (close[i] >= high_roll[i] or
                close[i] >= entry_price + 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout + weekly pivot direction + volume
            long_breakout = close[i] > high_roll[i-1]  # Break above previous period's high
            short_breakout = close[i] < low_roll[i-1]  # Break below previous period's low
            
            # Weekly pivot direction: above pivot = bullish bias, below = bearish bias
            above_weekly_pivot = close[i] > weekly_pivot_aligned[i]
            below_weekly_pivot = close[i] < weekly_pivot_aligned[i]
            
            # Volume confirmation: current volume > 1.5x average
            vol_confirm = volume[i] > (1.5 * vol_ma[i])
            
            if long_breakout and above_weekly_pivot and vol_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif short_breakout and below_weekly_pivot and vol_confirm:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals

</think>
#!/usr/bin/env python3
"""
6h Donchian Breakout with Weekly Pivot Direction and Volume Confirmation
Hypothesis: 6h Donchian(20) breakouts aligned with weekly pivot direction (above/below weekly pivot)
and volume confirmation capture institutional moves in both bull and bear markets.
Weekly pivot provides directional bias from higher timeframe, reducing false breakouts.
Volume ensures participation. Target: 50-150 total trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_donchian_breakout_weekly_pivot_volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Load 6h and weekly data (once before loop)
    df_6h = get_htf_data(prices, '6h')
    df_weekly = get_htf_data(prices, '1w')
    
    # 6h Donchian channels (20-period)
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    high_roll = pd.Series(high_6h).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low_6h).rolling(window=20, min_periods=20).min().values
    
    # Weekly pivot point (standard calculation: (H+L+C)/3)
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    close_weekly = df_weekly['close'].values
    weekly_pivot = (high_weekly + low_weekly + close_weekly) / 3.0
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_weekly, weekly_pivot)
    
    # 6h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 6h volume filter (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # 6h ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = 20  # For Donchian(20)
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits
        if position == 1:  # long position
            # Exit: price breaks below lower Donchian OR stoploss
            if (close[i] <= low_roll[i] or
                close[i] <= entry_price - 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price breaks above upper Donchian OR stoploss
            if (close[i] >= high_roll[i] or
                close[i] >= entry_price + 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout + weekly pivot direction + volume
            long_breakout = close[i] > high_roll[i-1]  # Break above previous period's high
            short_breakout = close[i] < low_roll[i-1]  # Break below previous period's low
            
            # Weekly pivot direction: above pivot = bullish bias, below = bearish bias
            above_weekly_pivot = close[i] > weekly_pivot_aligned[i]
            below_weekly_pivot = close[i] < weekly_pivot_aligned[i]
            
            # Volume confirmation: current volume > 1.5x average
            vol_confirm = volume[i] > (1.5 * vol_ma[i])
            
            if long_breakout and above_weekly_pivot and vol_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif short_breakout and below_weekly_pivot and vol_confirm:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals

</think>
#!/usr/bin/env python3
"""
6h Donchian Breakout with Weekly Pivot Direction and Volume Confirmation
Hypothesis: 6h Donchian(20) breakouts aligned with weekly pivot direction (above/below weekly pivot)
and volume confirmation capture institutional moves in both bull and bear markets.
Weekly pivot provides directional bias from higher timeframe, reducing false breakouts.
Volume ensures participation. Target: 50-150 total trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_donchian_breakout_weekly_pivot_volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Load 6h and weekly data (once before loop)
    df_6h = get_htf_data(prices, '6h')
    df_weekly = get_htf_data(prices, '1w')
    
    # 6h Donchian channels (20-period)
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    high_roll = pd.Series(high_6h).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low_6h).rolling(window=20, min_periods=20).min().values
    
    # Weekly pivot point (standard