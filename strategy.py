#!/usr/bin/env python3
"""
Experiment #9575: 6h Donchian Breakout + Weekly Pivot Direction + Volume Confirmation.
Hypothesis: Donchian(20) breakouts combined with weekly pivot direction provide high-probability
trend continuation signals in both bull and bear markets. Volume confirmation filters false breakouts.
Targets 100-200 total trades over 4 years (25-50/year) for optimal balance.
Works in bull markets (breakouts above weekly pivot) and bear markets (breakdowns below weekly pivot).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_9575_6h_donchian_breakout_weekly_pivot_volume_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
VOLUME_SPIKE_MULTIPLIER = 2.0
WEEKLY_PIVOT_METHOD = 'standard'  # standard pivot from weekly OHLC
SIGNAL_SIZE = 0.28
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def calculate_donchian_channels(high, low, period):
    """Calculate Donchian channels: upper = max(high, period), lower = min(low, period)"""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def calculate_weekly_pivot(high, low, close):
    """
    Calculate weekly pivot points (standard method)
    Pivot = (High + Low + Close) / 3
    R1 = (2 * Pivot) - Low
    S1 = (2 * Pivot) - High
    R2 = Pivot + (High - Low)
    S2 = Pivot - (High - Low)
    """
    pivot = (high + low + close) / 3.0
    r1 = (2 * pivot) - low
    s1 = (2 * pivot) - high
    r2 = pivot + (high - low)
    s2 = pivot - (high - low)
    return pivot, r1, r2, s1, s2

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1w for weekly pivot calculation)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot levels
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    pivot_1w, r1_1w, r2_1w, s1_1w, s2_1w = calculate_weekly_pivot(high_1w, low_1w, close_1w)
    
    # Align weekly levels to 6h timeframe
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    r2_1w_aligned = align_htf_to_ltf(prices, df_1w, r2_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    s2_1w_aligned = align_htf_to_ltf(prices, df_1w, s2_1w)
    
    # Calculate LTF indicators (6h)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels
    donchian_upper, donchian_lower = calculate_donchian_channels(high, low, DONCHIAN_PERIOD)
    
    # Volume moving average for spike detection
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for risk management
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, 20) + 1
    
    for i in range(start, n):
        # Skip if weekly pivot data not available
        if np.isnan(pivot_1w_aligned[i]) or np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        # Check stoploss
        if position == 1:  # long position
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # short position
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Volume spike confirmation
        volume_spike = volume[i] > (volume_ma[i] * VOLUME_SPIKE_MULTIPLIER) if not np.isnan(volume_ma[i]) else False
        
        # Breakout conditions with weekly pivot filter
        # Long: price breaks above Donchian upper AND above weekly pivot (bullish bias)
        donchian_breakout_long = close[i] > donchian_upper[i]
        weekly_pivot_bullish = close[i] > pivot_1w_aligned[i]  # Above weekly pivot = bullish bias
        long_entry = donchian_breakout_long and volume_spike and weekly_pivot_bullish
        
        # Short: price breaks below Donchian lower AND below weekly pivot (bearish bias)
        donchian_breakout_short = close[i] < donchian_lower[i]
        weekly_pivot_bearish = close[i] < pivot_1w_aligned[i]  # Below weekly pivot = bearish bias
        short_entry = donchian_breakout_short and volume_spike and weekly_pivot_bearish
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif short_entry:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals

</think>
#!/usr/bin/env python3
"""
Experiment #9575: 6h Donchian Breakout + Weekly Pivot Direction + Volume Confirmation.
Hypothesis: Donchian(20) breakouts combined with weekly pivot direction provide high-probability
trend continuation signals in both bull and bear markets. Volume confirmation filters false breakouts.
Targets 100-200 total trades over 4 years (25-50/year) for optimal balance.
Works in bull markets (breakouts above weekly pivot) and bear markets (breakdowns below weekly pivot).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_9575_6h_donchian_breakout_weekly_pivot_volume_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
VOLUME_SPIKE_MULTIPLIER = 2.0
WEEKLY_PIVOT_METHOD = 'standard'  # standard pivot from weekly OHLC
SIGNAL_SIZE = 0.28
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def calculate_donchian_channels(high, low, period):
    """Calculate Donchian channels: upper = max(high, period), lower = min(low, period)"""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def calculate_weekly_pivot(high, low, close):
    """
    Calculate weekly pivot points (standard method)
    Pivot = (High + Low + Close) / 3
    R1 = (2 * Pivot) - Low
    S1 = (2 * Pivot) - High
    R2 = Pivot + (High - Low)
    S2 = Pivot - (High - Low)
    """
    pivot = (high + low + close) / 3.0
    r1 = (2 * pivot) - low
    s1 = (2 * pivot) - high
    r2 = pivot + (high - low)
    s2 = pivot - (high - low)
    return pivot, r1, r2, s1, s2

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1w for weekly pivot calculation)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot levels
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    pivot_1w, r1_1w, r2_1w, s1_1w, s2_1w = calculate_weekly_pivot(high_1w, low_1w, close_1w)
    
    # Align weekly levels to 6h timeframe
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    r2_1w_aligned = align_htf_to_ltf(prices, df_1w, r2_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    s2_1w_aligned = align_htf_to_ltf(prices, df_1w, s2_1w)
    
    # Calculate LTF indicators (6h)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels
    donchian_upper, donchian_lower = calculate_donchian_channels(high, low, DONCHIAN_PERIOD)
    
    # Volume moving average for spike detection
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for risk management
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, 20) + 1
    
    for i in range(start, n):
        # Skip if weekly pivot data not available
        if np.isnan(pivot_1w_aligned[i]) or np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        # Check stoploss
        if position == 1:  # long position
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # short position
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Volume spike confirmation
        volume_spike = volume[i] > (volume_ma[i] * VOLUME_SPIKE_MULTIPLIER) if not np.isnan(volume_ma[i]) else False
        
        # Breakout conditions with weekly pivot filter
        # Long: price breaks above Donchian upper AND above weekly pivot (bullish bias)
        donchian_breakout_long = close[i] > donchian_upper[i]
        weekly_pivot_bullish = close[i] > pivot_1w_aligned[i]  # Above weekly pivot = bullish bias
        long_entry = donchian_breakout_long and volume_spike and weekly_pivot_bullish
        
        # Short: price breaks below Donchian lower AND below weekly pivot (bearish bias)
        donchian_breakout_short = close[i] < donchian_lower[i]
        weekly_pivot_bearish = close[i] < pivot_1w_aligned[i]  # Below weekly pivot = bearish bias
        short_entry = donchian_breakout_short and volume_spike and weekly_pivot_bearish
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif short_entry:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals