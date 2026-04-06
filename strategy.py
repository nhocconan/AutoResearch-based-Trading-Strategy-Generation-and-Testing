#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_13970_1d_donchian20_1w_pivot_vol_v1"
timeframe = "1d"
leverage = 1.0

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def calculate_pivot(high, low, close):
    """Calculate pivot point and support/resistance levels"""
    pivot = (high + low + close) / 3.0
    r1 = 2 * pivot - low
    s1 = 2 * pivot - high
    r2 = pivot + (high - low)
    s2 = pivot - (high - low)
    r3 = high + 2 * (pivot - low)
    s3 = low - 2 * (high - pivot)
    return pivot, r1, r2, r3, s1, s2, s3

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data for pivot calculation ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot levels (using prior week's data)
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    pivots, r1, r2, r3, s1, s2, s3 = calculate_pivot(weekly_high, weekly_low, weekly_close)
    
    # Align weekly pivot to daily timeframe (use prior week's pivot for bias)
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivots)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    
    # Daily data for Donchian, ATR, and volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    donchian_upper = np.full(n, np.nan)
    donchian_lower = np.full(n, np.nan)
    for i in range(20-1, n):
        donchian_upper[i] = np.max(high[i-20+1:i+1])
        donchian_lower[i] = np.min(low[i-20+1:i+1])
    
    # ATR for stop loss
    atr = calculate_atr(high, low, close, 14)
    
    # Volume confirmation (20-day average)
    volume_ma = np.full(n, np.nan)
    for i in range(20-1, n):
        volume_ma[i] = np.mean(volume[i-20+1:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(50, 20) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(pivot_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or \
           np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or \
           np.isnan(volume_ma[i]) or np.isnan(atr[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check stops
        if position == 1:  # long position
            # Check stop loss
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # short position
            # Check stop loss
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Determine bias from weekly pivot (price vs pivot level)
        bullish_bias = close[i] > pivot_aligned[i]  # price above weekly pivot = bullish bias
        bearish_bias = close[i] < pivot_aligned[i]  # price below weekly pivot = bearish bias
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * 1.3)
        
        # Donchian breakout signals
        breakout_up = close[i] > donchian_upper[i-1]  # break above previous upper band
        breakout_down = close[i] < donchian_lower[i-1]  # break below previous lower band
        
        # Entry signals - only in direction of weekly bias
        long_signal = bullish_bias and volume_ok and breakout_up
        short_signal = bearish_bias and volume_ok and breakout_down
        
        # Generate signals
        if position == 0:
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (2.0 * atr[i])
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (2.0 * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long on Donchian breakdown or bias change to bearish
            if close[i] < donchian_lower[i] or not bullish_bias:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short on Donchian breakout or bias change to bullish
            if close[i] > donchian_upper[i] or not bearish_bias:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

</think>

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_13970_1d_donchian20_1w_pivot_vol_v1"
timeframe = "1d"
leverage = 1.0

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def calculate_pivot(high, low, close):
    """Calculate pivot point and support/resistance levels"""
    pivot = (high + low + close) / 3.0
    r1 = 2 * pivot - low
    s1 = 2 * pivot - high
    r2 = pivot + (high - low)
    s2 = pivot - (high - low)
    r3 = high + 2 * (pivot - low)
    s3 = low - 2 * (high - pivot)
    return pivot, r1, r2, r3, s1, s2, s3

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data for pivot calculation ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot levels (using prior week's data)
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    pivots, r1, r2, r3, s1, s2, s3 = calculate_pivot(weekly_high, weekly_low, weekly_close)
    
    # Align weekly pivot to daily timeframe (use prior week's pivot for bias)
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivots)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    
    # Daily data for Donchian, ATR, and volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    donchian_upper = np.full(n, np.nan)
    donchian_lower = np.full(n, np.nan)
    for i in range(20-1, n):
        donchian_upper[i] = np.max(high[i-20+1:i+1])
        donchian_lower[i] = np.min(low[i-20+1:i+1])
    
    # ATR for stop loss
    atr = calculate_atr(high, low, close, 14)
    
    # Volume confirmation (20-day average)
    volume_ma = np.full(n, np.nan)
    for i in range(20-1, n):
        volume_ma[i] = np.mean(volume[i-20+1:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(50, 20) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(pivot_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or \
           np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or \
           np.isnan(volume_ma[i]) or np.isnan(atr[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check stops
        if position == 1:  # long position
            # Check stop loss
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # short position
            # Check stop loss
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Determine bias from weekly pivot (price vs pivot level)
        bullish_bias = close[i] > pivot_aligned[i]  # price above weekly pivot = bullish bias
        bearish_bias = close[i] < pivot_aligned[i]  # price below weekly pivot = bearish bias
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * 1.3)
        
        # Donchian breakout signals
        breakout_up = close[i] > donchian_upper[i-1]  # break above previous upper band
        breakout_down = close[i] < donchian_lower[i-1]  # break below previous lower band
        
        # Entry signals - only in direction of weekly bias
        long_signal = bullish_bias and volume_ok and breakout_up
        short_signal = bearish_bias and volume_ok and breakout_down
        
        # Generate signals
        if position == 0:
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (2.0 * atr[i])
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (2.0 * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long on Donchian breakdown or bias change to bearish
            if close[i] < donchian_lower[i] or not bullish_bias:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short on Donchian breakout or bias change to bullish
            if close[i] > donchian_upper[i] or not bearish_bias:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

</think>