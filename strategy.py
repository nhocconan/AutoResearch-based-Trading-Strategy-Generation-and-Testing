#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 12h Supertrend for regime filter and 6h ATR breakout for entry.
# Supertrend identifies strong trends (bull/bear) while avoiding chop. ATR breakout captures momentum in direction of trend.
# Works in bull (long breakouts in uptrend) and bear (short breakouts in downtrend) regimes.
# Target: 50-150 trades over 4 years (12-37/year). Size: 0.25.

name = "6h_Supertrend12h_ATRBreakout_Regime_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Supertrend (regime filter)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 10:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h ATR(10) for Supertrend
    tr1_12h = high_12h - low_12h
    tr2_12h = np.abs(high_12h - np.roll(close_12h, 1))
    tr3_12h = np.abs(low_12h - np.roll(close_12h, 1))
    tr_12h = np.maximum(tr1_12h, np.maximum(tr2_12h, tr3_12h))
    tr_12h[0] = 0
    atr_12h = pd.Series(tr_12h).rolling(window=10, min_periods=10).mean().values
    
    # Supertrend parameters
    atr_mult = 3.0
    upper_band_12h = (high_12h + low_12h) / 2 + atr_mult * atr_12h
    lower_band_12h = (high_12h + low_12h) / 2 - atr_mult * atr_12h
    
    # Initialize Supertrend
    supertrend_12h = np.full_like(close_12h, np.nan, dtype=float)
    direction_12h = np.full_like(close_12h, np.nan, dtype=float)  # 1 for uptrend, -1 for downtrend
    
    for i in range(len(close_12h)):
        if i == 0:
            supertrend_12h[i] = upper_band_12h[i]
            direction_12h[i] = 1
        else:
            if close_12h[i-1] > supertrend_12h[i-1]:
                supertrend_12h[i] = max(lower_band_12h[i], supertrend_12h[i-1])
                direction_12h[i] = 1
            else:
                supertrend_12h[i] = min(upper_band_12h[i], supertrend_12h[i-1])
                direction_12h[i] = -1
    
    # Align Supertrend direction to 6h
    supertrend_12h_aligned = align_htf_to_ltf(prices, df_12h, supertrend_12h)
    direction_12h_aligned = align_htf_to_ltf(prices, df_12h, direction_12h)
    
    # 6h ATR(14) for breakout threshold
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = 0
    atr_6h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 6h Donchian(20) for breakout levels
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 14)
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(direction_12h_aligned[i]) or 
            np.isnan(atr_6h[i]) or
            np.isnan(highest_high_20[i]) or
            np.isnan(lowest_low_20[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: 12h Supertrend direction
        uptrend_regime = direction_12h_aligned[i] == 1
        downtrend_regime = direction_12h_aligned[i] == -1
        
        # Breakout conditions
        long_breakout = close[i] > highest_high_20[i]
        short_breakout = close[i] < lowest_low_20[i]
        
        # ATR breakout confirmation: breakout must exceed 0.5 * ATR
        long_atr_confirm = (close[i] - highest_high_20[i]) > 0.5 * atr_6h[i]
        short_atr_confirm = (lowest_low_20[i] - close[i]) > 0.5 * atr_6h[i]
        
        long_entry = uptrend_regime and long_breakout and long_atr_confirm
        short_entry = downtrend_regime and short_breakout and short_atr_confirm
        
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

# Hypothesis: 6h strategy using 12h Supertrend for regime filter and 6h ATR breakout for entry.
# Supertrend identifies strong trends (bull/bear) while avoiding chop. ATR breakout captures momentum in direction of trend.
# Works in bull (long breakouts in uptrend) and bear (short breakouts in downtrend) regimes.
# Target: 50-150 trades over 4 years (12-37/year). Size: 0.25.

name = "6h_Supertrend12h_ATRBreakout_Regime_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Supertrend (regime filter)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 10:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h ATR(10) for Supertrend
    tr1_12h = high_12h - low_12h
    tr2_12h = np.abs(high_12h - np.roll(close_12h, 1))
    tr3_12h = np.abs(low_12h - np.roll(close_12h, 1))
    tr_12h = np.maximum(tr1_12h, np.maximum(tr2_12h, tr3_12h))
    tr_12h[0] = 0
    atr_12h = pd.Series(tr_12h).rolling(window=10, min_periods=10).mean().values
    
    # Supertrend parameters
    atr_mult = 3.0
    upper_band_12h = (high_12h + low_12h) / 2 + atr_mult * atr_12h
    lower_band_12h = (high_12h + low_12h) / 2 - atr_mult * atr_12h
    
    # Initialize Supertrend
    supertrend_12h = np.full_like(close_12h, np.nan, dtype=float)
    direction_12h = np.full_like(close_12h, np.nan, dtype=float)  # 1 for uptrend, -1 for downtrend
    
    for i in range(len(close_12h)):
        if i == 0:
            supertrend_12h[i] = upper_band_12h[i]
            direction_12h[i] = 1
        else:
            if close_12h[i-1] > supertrend_12h[i-1]:
                supertrend_12h[i] = max(lower_band_12h[i], supertrend_12h[i-1])
                direction_12h[i] = 1
            else:
                supertrend_12h[i] = min(upper_band_12h[i], supertrend_12h[i-1])
                direction_12h[i] = -1
    
    # Align Supertrend direction to 6h
    supertrend_12h_aligned = align_htf_to_ltf(prices, df_12h, supertrend_12h)
    direction_12h_aligned = align_htf_to_ltf(prices, df_12h, direction_12h)
    
    # 6h ATR(14) for breakout threshold
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = 0
    atr_6h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 6h Donchian(20) for breakout levels
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 14)
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(direction_12h_aligned[i]) or 
            np.isnan(atr_6h[i]) or
            np.isnan(highest_high_20[i]) or
            np.isnan(lowest_low_20[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: 12h Supertrend direction
        uptrend_regime = direction_12h_aligned[i] == 1
        downtrend_regime = direction_12h_aligned[i] == -1
        
        # Breakout conditions
        long_breakout = close[i] > highest_high_20[i]
        short_breakout = close[i] < lowest_low_20[i]
        
        # ATR breakout confirmation: breakout must exceed 0.5 * ATR
        long_atr_confirm = (close[i] - highest_high_20[i]) > 0.5 * atr_6h[i]
        short_atr_confirm = (lowest_low_20[i] - close[i]) > 0.5 * atr_6h[i]
        
        long_entry = uptrend_regime and long_breakout and long_atr_confirm
        short_entry = downtrend_regime and short_breakout and short_atr_confirm
        
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

name = "6h_Supertrend12h_ATRBreakout_Regime_v1"
timeframe = "6h"
leverage = 1.0