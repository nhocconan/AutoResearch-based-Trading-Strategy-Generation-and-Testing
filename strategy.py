#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 12h KAMA trend filter and 4h Donchian(20) breakout with volume confirmation.
# KAMA adapts to market noise, reducing whipsaws in choppy markets. Donchian breakouts capture momentum in direction of trend.
# Volume confirmation ensures breakouts have participation. Works in bull (long breakouts in uptrend) and bear (short breakouts in downtrend).
# Target: 75-200 trades over 4 years (19-50/year). Size: 0.25.

name = "4h_KAMA12h_Donchian20_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for KAMA (trend filter)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate 12h KAMA(30,2,30)
    # Efficiency Ratio (ER)
    change_12h = np.abs(np.diff(close_12h, prepend=close_12h[0]))
    volatility_12h = np.sum(np.abs(np.diff(close_12h, prepend=close_12h[0])), axis=0) if False else None  # placeholder for correct calculation
    # Recalculate volatility correctly
    volatility_12h = np.zeros_like(close_12h)
    for i in range(1, len(close_12h)):
        volatility_12h[i] = volatility_12h[i-1] + np.abs(close_12h[i] - close_12h[i-1])
    er_12h = np.zeros_like(close_12h)
    for i in range(30, len(close_12h)):
        if volatility_12h[i] != 0:
            er_12h[i] = np.abs(close_12h[i] - close_12h[i-30]) / volatility_12h[i]
        else:
            er_12h[i] = 0
    # Smoothing constants
    fast_sc = 2 / (2 + 1)
    slow_sc = 2 / (30 + 1)
    sc_12h = (er_12h * (fast_sc - slow_sc) + slow_sc) ** 2
    # Initialize KAMA
    kama_12h = np.full_like(close_12h, np.nan, dtype=float)
    kama_12h[29] = close_12h[29]  # seed
    for i in range(30, len(close_12h)):
        kama_12h[i] = kama_12h[i-1] + sc_12h[i] * (close_12h[i] - kama_12h[i-1])
    
    # Align KAMA to 4h
    kama_12h_aligned = align_htf_to_ltf(prices, df_12h, kama_12h)
    
    # 4h ATR(14) for stoploss (not used in signals but for regime)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = 0
    atr_4h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 4h Donchian(20) for breakout levels
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 4h volume average for confirmation
    volume_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)  # KAMA seed + Donchian
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(kama_12h_aligned[i]) or 
            np.isnan(highest_high_20[i]) or
            np.isnan(lowest_low_20[i]) or
            np.isnan(volume_avg_20[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price relative to 12h KAMA
        uptrend_regime = close[i] > kama_12h_aligned[i]
        downtrend_regime = close[i] < kama_12h_aligned[i]
        
        # Breakout conditions
        long_breakout = close[i] > highest_high_20[i]
        short_breakout = close[i] < lowest_low_20[i]
        
        # Volume confirmation: volume > 1.5 * average
        volume_confirm = volume[i] > 1.5 * volume_avg_20[i]
        
        long_entry = uptrend_regime and long_breakout and volume_confirm
        short_entry = downtrend_regime and short_breakout and volume_confirm
        
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

# Hypothesis: 4h strategy using 12h KAMA trend filter and 4h Donchian(20) breakout with volume confirmation.
# KAMA adapts to market noise, reducing whipsaws in choppy markets. Donchian breakouts capture momentum in direction of trend.
# Volume confirmation ensures breakouts have participation. Works in bull (long breakouts in uptrend) and bear (short breakouts in downtrend).
# Target: 75-200 trades over 4 years (19-50/year). Size: 0.25.

name = "4h_KAMA12h_Donchian20_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for KAMA (trend filter)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate 12h KAMA(30,2,30)
    # Efficiency Ratio (ER)
    change_12h = np.abs(np.diff(close_12h, prepend=close_12h[0]))
    volatility_12h = np.sum(np.abs(np.diff(close_12h, prepend=close_12h[0])), axis=0) if False else None  # placeholder for correct calculation
    # Recalculate volatility correctly
    volatility_12h = np.zeros_like(close_12h)
    for i in range(1, len(close_12h)):
        volatility_12h[i] = volatility_12h[i-1] + np.abs(close_12h[i] - close_12h[i-1])
    er_12h = np.zeros_like(close_12h)
    for i in range(30, len(close_12h)):
        if volatility_12h[i] != 0:
            er_12h[i] = np.abs(close_12h[i] - close_12h[i-30]) / volatility_12h[i]
        else:
            er_12h[i] = 0
    # Smoothing constants
    fast_sc = 2 / (2 + 1)
    slow_sc = 2 / (30 + 1)
    sc_12h = (er_12h * (fast_sc - slow_sc) + slow_sc) ** 2
    # Initialize KAMA
    kama_12h = np.full_like(close_12h, np.nan, dtype=float)
    kama_12h[29] = close_12h[29]  # seed
    for i in range(30, len(close_12h)):
        kama_12h[i] = kama_12h[i-1] + sc_12h[i] * (close_12h[i] - kama_12h[i-1])
    
    # Align KAMA to 4h
    kama_12h_aligned = align_htf_to_ltf(prices, df_12h, kama_12h)
    
    # 4h ATR(14) for stoploss (not used in signals but for regime)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = 0
    atr_4h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 4h Donchian(20) for breakout levels
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 4h volume average for confirmation
    volume_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)  # KAMA seed + Donchian
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(kama_12h_aligned[i]) or 
            np.isnan(highest_high_20[i]) or
            np.isnan(lowest_low_20[i]) or
            np.isnan(volume_avg_20[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price relative to 12h KAMA
        uptrend_regime = close[i] > kama_12h_aligned[i]
        downtrend_regime = close[i] < kama_12h_aligned[i]
        
        # Breakout conditions
        long_breakout = close[i] > highest_high_20[i]
        short_breakout = close[i] < lowest_low_20[i]
        
        # Volume confirmation: volume > 1.5 * average
        volume_confirm = volume[i] > 1.5 * volume_avg_20[i]
        
        long_entry = uptrend_regime and long_breakout and volume_confirm
        short_entry = downtrend_regime and short_breakout and volume_confirm
        
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

name = "4h_KAMA12h_Donchian20_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0