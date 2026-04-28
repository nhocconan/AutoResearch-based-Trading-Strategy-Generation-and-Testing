#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using 1w Supertrend for regime filter and Donchian(20) breakout for entry.
# Supertrend on weekly timeframe identifies strong bull/bear regimes while filtering chop.
# Donchian breakout captures momentum in direction of weekly trend.
# Works in bull (long breakouts in uptrend) and bear (short breakouts in downtrend) regimes.
# Target: 30-100 trades over 4 years (7-25/year). Size: 0.25.

name = "1d_Supertrend1w_Donchian20_Breakout_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for Supertrend (regime filter)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1w ATR(10) for Supertrend
    tr1_1w = high_1w - low_1w
    tr2_1w = np.abs(high_1w - np.roll(close_1w, 1))
    tr3_1w = np.abs(low_1w - np.roll(close_1w, 1))
    tr_1w = np.maximum(tr1_1w, np.maximum(tr2_1w, tr3_1w))
    tr_1w[0] = 0
    atr_1w = pd.Series(tr_1w).rolling(window=10, min_periods=10).mean().values
    
    # Supertrend parameters
    atr_mult = 3.0
    upper_band_1w = (high_1w + low_1w) / 2 + atr_mult * atr_1w
    lower_band_1w = (high_1w + low_1w) / 2 - atr_mult * atr_1w
    
    # Initialize Supertrend
    supertrend_1w = np.full_like(close_1w, np.nan, dtype=float)
    direction_1w = np.full_like(close_1w, np.nan, dtype=float)  # 1 for uptrend, -1 for downtrend
    
    for i in range(len(close_1w)):
        if i == 0:
            supertrend_1w[i] = upper_band_1w[i]
            direction_1w[i] = 1
        else:
            if close_1w[i-1] > supertrend_1w[i-1]:
                supertrend_1w[i] = max(lower_band_1w[i], supertrend_1w[i-1])
                direction_1w[i] = 1
            else:
                supertrend_1w[i] = min(upper_band_1w[i], supertrend_1w[i-1])
                direction_1w[i] = -1
    
    # Align Supertrend direction to 1d
    supertrend_1w_aligned = align_htf_to_ltf(prices, df_1w, supertrend_1w)
    direction_1w_aligned = align_htf_to_ltf(prices, df_1w, direction_1w)
    
    # 1d Donchian(20) for breakout levels
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(direction_1w_aligned[i]) or
            np.isnan(highest_high_20[i]) or
            np.isnan(lowest_low_20[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: 1w Supertrend direction
        uptrend_regime = direction_1w_aligned[i] == 1
        downtrend_regime = direction_1w_aligned[i] == -1
        
        # Breakout conditions
        long_breakout = close[i] > highest_high_20[i]
        short_breakout = close[i] < lowest_low_20[i]
        
        long_entry = uptrend_regime and long_breakout
        short_entry = downtrend_regime and short_breakout
        
        # Exit: opposite Donchian breakout (10-bar for faster exit)
        highest_high_10 = pd.Series(high).rolling(window=10, min_periods=10).max().values
        lowest_low_10 = pd.Series(low).rolling(window=10, min_periods=10).min().values
        long_exit = close[i] < highest_high_10[i]
        short_exit = close[i] > lowest_low_10[i]
        
        # Handle entries and exits
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif (position == 1 and long_exit) or (position == -1 and short_exit):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using 1w Supertrend for regime filter and Donchian(20) breakout for entry.
# Supertrend on weekly timeframe identifies strong bull/bear regimes while filtering chop.
# Donchian breakout captures momentum in direction of weekly trend.
# Works in bull (long breakouts in uptrend) and bear (short breakouts in downtrend) regimes.
# Target: 30-100 trades over 4 years (7-25/year). Size: 0.25.

name = "1d_Supertrend1w_Donchian20_Breakout_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for Supertrend (regime filter)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1w ATR(10) for Supertrend
    tr1_1w = high_1w - low_1w
    tr2_1w = np.abs(high_1w - np.roll(close_1w, 1))
    tr3_1w = np.abs(low_1w - np.roll(close_1w, 1))
    tr_1w = np.maximum(tr1_1w, np.maximum(tr2_1w, tr3_1w))
    tr_1w[0] = 0
    atr_1w = pd.Series(tr_1w).rolling(window=10, min_periods=10).mean().values
    
    # Supertrend parameters
    atr_mult = 3.0
    upper_band_1w = (high_1w + low_1w) / 2 + atr_mult * atr_1w
    lower_band_1w = (high_1w + low_1w) / 2 - atr_mult * atr_1w
    
    # Initialize Supertrend
    supertrend_1w = np.full_like(close_1w, np.nan, dtype=float)
    direction_1w = np.full_like(close_1w, np.nan, dtype=float)  # 1 for uptrend, -1 for downtrend
    
    for i in range(len(close_1w)):
        if i == 0:
            supertrend_1w[i] = upper_band_1w[i]
            direction_1w[i] = 1
        else:
            if close_1w[i-1] > supertrend_1w[i-1]:
                supertrend_1w[i] = max(lower_band_1w[i], supertrend_1w[i-1])
                direction_1w[i] = 1
            else:
                supertrend_1w[i] = min(upper_band_1w[i], supertrend_1w[i-1])
                direction_1w[i] = -1
    
    # Align Supertrend direction to 1d
    supertrend_1w_aligned = align_htf_to_ltf(prices, df_1w, supertrend_1w)
    direction_1w_aligned = align_htf_to_ltf(prices, df_1w, direction_1w)
    
    # 1d Donchian(20) for breakout levels
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(direction_1w_aligned[i]) or
            np.isnan(highest_high_20[i]) or
            np.isnan(lowest_low_20[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: 1w Supertrend direction
        uptrend_regime = direction_1w_aligned[i] == 1
        downtrend_regime = direction_1w_aligned[i] == -1
        
        # Breakout conditions
        long_breakout = close[i] > highest_high_20[i]
        short_breakout = close[i] < lowest_low_20[i]
        
        long_entry = uptrend_regime and long_breakout
        short_entry = downtrend_regime and short_breakout
        
        # Exit: opposite Donchian breakout (10-bar for faster exit)
        highest_high_10 = pd.Series(high).rolling(window=10, min_periods=10).max().values
        lowest_low_10 = pd.Series(low).rolling(window=10, min_periods=10).min().values
        long_exit = close[i] < highest_high_10[i]
        short_exit = close[i] > lowest_low_10[i]
        
        # Handle entries and exits
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif (position == 1 and long_exit) or (position == -1 and short_exit):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_Supertrend1w_Donchian20_Breakout_v1"
timeframe = "1d"
leverage = 1.0