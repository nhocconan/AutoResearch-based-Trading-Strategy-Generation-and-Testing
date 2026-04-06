#!/usr/bin/env python3
"""
12h Donchian(20) Breakout + Volume Confirmation + 1d Trend Filter
Hypothesis: Donchian breakouts capture strong momentum moves. Volume confirms institutional participation.
1d EMA50 filters trend direction to avoid counter-trend trades. Works in both bull (breakout long) and bear (breakdown short).
Target: 50-150 total trades over 4 years (12-37/year) with 0.25 position size to manage drawdown.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_donchian20_1d_trend_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_prev = np.roll(ema50_1d, 1)
    ema50_1d_prev[0] = ema50_1d[0]
    ema50_rising = ema50_1d > ema50_1d_prev
    ema50_falling = ema50_1d < ema50_1d_prev
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    ema50_rising_aligned = align_htf_to_ltf(prices, df_1d, ema50_rising)
    ema50_falling_aligned = align_htf_to_ltf(prices, df_1d, ema50_falling)
    
    # 12h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    def calculate_donchian(high_arr, low_arr, period):
        upper = np.full_like(high_arr, np.nan)
        lower = np.full_like(low_arr, np.nan)
        for i in range(period-1, len(high_arr)):
            upper[i] = np.max(high_arr[i-period+1:i+1])
            lower[i] = np.min(low_arr[i-period+1:i+1])
        return upper, lower
    
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    
    # Volume filter: 20-period EMA
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = 50  # For Donchian(20) and EMA50
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(vol_ema[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(ema50_rising_aligned[i]) or np.isnan(ema50_falling_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: reverse signal or stoploss
        if position == 1:  # long position
            # Exit: price breaks below Donchian lower OR stoploss
            atr_proxy = (high[i] - low[i])  # Simple range as ATR proxy
            if (close[i] <= donchian_lower[i] or 
                close[i] <= entry_price - 2.0 * atr_proxy):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price breaks above Donchian upper OR stoploss
            atr_proxy = (high[i] - low[i])
            if (close[i] >= donchian_upper[i] or 
                close[i] >= entry_price + 2.0 * atr_proxy):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout + trend + volume
            bull_breakout = close[i] > donchian_upper[i]
            bear_breakdown = close[i] < donchian_lower[i]
            
            bull_entry = bull_breakout and ema50_rising_aligned[i] and volume[i] > vol_ema[i] * 1.5
            bear_entry = bear_breakdown and ema50_falling_aligned[i] and volume[i] > vol_ema[i] * 1.5
            
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
12h Donchian(20) Breakout + Volume Confirmation + 1d Trend Filter
Hypothesis: Donchian breakouts capture strong momentum moves. Volume confirms institutional participation.
1d EMA50 filters trend direction to avoid counter-trend trades. Works in both bull (breakout long) and bear (breakdown short).
Target: 50-150 total trades over 4 years (12-37/year) with 0.25 position size to manage drawdown.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_donchian20_1d_trend_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_prev = np.roll(ema50_1d, 1)
    ema50_1d_prev[0] = ema50_1d[0]
    ema50_rising = ema50_1d > ema50_1d_prev
    ema50_falling = ema50_1d < ema50_1d_prev
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    ema50_rising_aligned = align_htf_to_ltf(prices, df_1d, ema50_rising)
    ema50_falling_aligned = align_htf_to_ltf(prices, df_1d, ema50_falling)
    
    # 12h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    def calculate_donchian(high_arr, low_arr, period):
        upper = np.full_like(high_arr, np.nan)
        lower = np.full_like(low_arr, np.nan)
        for i in range(period-1, len(high_arr)):
            upper[i] = np.max(high_arr[i-period+1:i+1])
            lower[i] = np.min(low_arr[i-period+1:i+1])
        return upper, lower
    
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    
    # Volume filter: 20-period EMA
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = 50  # For Donchian(20) and EMA50
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(vol_ema[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(ema50_rising_aligned[i]) or np.isnan(ema50_falling_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: reverse signal or stoploss
        if position == 1:  # long position
            # Exit: price breaks below Donchian lower OR stoploss
            atr_proxy = (high[i] - low[i])  # Simple range as ATR proxy
            if (close[i] <= donchian_lower[i] or 
                close[i] <= entry_price - 2.0 * atr_proxy):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price breaks above Donchian upper OR stoploss
            atr_proxy = (high[i] - low[i])
            if (close[i] >= donchian_upper[i] or 
                close[i] >= entry_price + 2.0 * atr_proxy):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout + trend + volume
            bull_breakout = close[i] > donchian_upper[i]
            bear_breakdown = close[i] < donchian_lower[i]
            
            bull_entry = bull_breakout and ema50_rising_aligned[i] and volume[i] > vol_ema[i] * 1.5
            bear_entry = bear_breakdown and ema50_falling_aligned[i] and volume[i] > vol_ema[i] * 1.5
            
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