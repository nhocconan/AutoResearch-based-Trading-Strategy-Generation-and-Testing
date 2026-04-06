#!/usr/bin/env python3
"""
1d Donchian(20) breakout with 1w trend filter and volume confirmation
Hypothesis: Donchian breakouts capture momentum, while weekly EMA200 filters trend direction to avoid counter-trend trades.
Volume confirms breakout strength. Works in bull markets (buy breakouts above upper band) and bear markets (sell breakdowns below lower band).
Target: 75-200 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_donchian20_1w_trend_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1w data for trend filter (once before loop)
    df_1w = get_htf_data(prices, '1w')
    
    # 1w EMA200 for trend filter
    close_1w = df_1w['close'].values
    ema200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1w_prev = np.roll(ema200_1w, 1)
    ema200_1w_prev[0] = ema200_1w[0]
    ema200_rising = ema200_1w > ema200_1w_prev
    ema200_falling = ema200_1w < ema200_1w_prev
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    ema200_rising_aligned = align_htf_to_ltf(prices, df_1w, ema200_rising)
    ema200_falling_aligned = align_htf_to_ltf(prices, df_1w, ema200_falling)
    
    # Daily data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian(20): upper = max(high, 20), lower = min(low, 20)
    # Using pandas rolling for simplicity and correctness
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    donchian_upper = high_s.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_s.rolling(window=20, min_periods=20).min().values
    
    # Volume filter: 20-period EMA
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = 200  # For EMA200
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(vol_ema[i]) or np.isnan(ema200_1w_aligned[i]) or 
            np.isnan(ema200_rising_aligned[i]) or np.isnan(ema200_falling_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: reverse signal or stoploss
        if position == 1:  # long position
            # Exit: price breaks below lower band OR stoploss
            if (close[i] <= donchian_lower[i] or 
                close[i] <= entry_price - 2.5 * (high[i] - low[i])):  # ATR proxy
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price breaks above upper band OR stoploss
            if (close[i] >= donchian_upper[i] or 
                close[i] >= entry_price + 2.5 * (high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout + trend + volume
            bull_breakout = close[i] > donchian_upper[i]
            bear_breakout = close[i] < donchian_lower[i]
            
            bull_entry = bull_breakout and ema200_rising_aligned[i] and volume[i] > vol_ema[i] * 1.5
            bear_entry = bear_breakout and ema200_falling_aligned[i] and volume[i] > vol_ema[i] * 1.5
            
            if bull_entry:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif bear_entry:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals

</think>

#!/usr/bin/env python3
"""
1d Donchian(20) breakout with 1w trend filter and volume confirmation
Hypothesis: Donchian breakouts capture momentum, while weekly EMA200 filters trend direction to avoid counter-trend trades.
Volume confirms breakout strength. Works in bull markets (buy breakouts above upper band) and bear markets (sell breakdowns below lower band).
Target: 75-200 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_donchian20_1w_trend_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1w data for trend filter (once before loop)
    df_1w = get_htf_data(prices, '1w')
    
    # 1w EMA200 for trend filter
    close_1w = df_1w['close'].values
    ema200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1w_prev = np.roll(ema200_1w, 1)
    ema200_1w_prev[0] = ema200_1w[0]
    ema200_rising = ema200_1w > ema200_1w_prev
    ema200_falling = ema200_1w < ema200_1w_prev
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    ema200_rising_aligned = align_htf_to_ltf(prices, df_1w, ema200_rising)
    ema200_falling_aligned = align_htf_to_ltf(prices, df_1w, ema200_falling)
    
    # Daily data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian(20): upper = max(high, 20), lower = min(low, 20)
    # Using pandas rolling for simplicity and correctness
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    donchian_upper = high_s.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_s.rolling(window=20, min_periods=20).min().values
    
    # Volume filter: 20-period EMA
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = 200  # For EMA200
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(vol_ema[i]) or np.isnan(ema200_1w_aligned[i]) or 
            np.isnan(ema200_rising_aligned[i]) or np.isnan(ema200_falling_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: reverse signal or stoploss
        if position == 1:  # long position
            # Exit: price breaks below lower band OR stoploss
            if (close[i] <= donchian_lower[i] or 
                close[i] <= entry_price - 2.5 * (high[i] - low[i])):  # ATR proxy
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price breaks above upper band OR stoploss
            if (close[i] >= donchian_upper[i] or 
                close[i] >= entry_price + 2.5 * (high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout + trend + volume
            bull_breakout = close[i] > donchian_upper[i]
            bear_breakout = close[i] < donchian_lower[i]
            
            bull_entry = bull_breakout and ema200_rising_aligned[i] and volume[i] > vol_ema[i] * 1.5
            bear_entry = bear_breakout and ema200_falling_aligned[i] and volume[i] > vol_ema[i] * 1.5
            
            if bull_entry:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif bear_entry:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals

</think>