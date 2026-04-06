#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout + 1d pivot direction + volume confirmation.
# In bull markets: buy when price breaks above Donchian(20) high and 1d pivot shows bullish bias (price > pivot).
# In bear markets: sell when price breaks below Donchian(20) low and 1d pivot shows bearish bias (price < pivot).
# Volume confirmation ensures institutional participation. This structure-based approach avoids whipsaws.
# Target: 50-150 total trades over 4 years = 12-37/year by using Donchian breakouts (natural low frequency)
# combined with pivot filter and volume confirmation.

name = "exp_13627_6h_donchian20_1d_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5

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
    tr[0] = tr1[0]  # First TR is just high-low
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def calculate_pivot(high, low, close):
    """Calculate standard pivot point: (H + L + C) / 3"""
    return (high + low + close) / 3.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for pivot filter ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d pivot and bias
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    pivot_1d = calculate_pivot(high_1d, low_1d, close_1d)
    bullish_bias = close_1d > pivot_1d  # price above pivot = bullish
    bearish_bias = close_1d < pivot_1d   # price below pivot = bearish
    bullish_bias_aligned = align_htf_to_ltf(prices, df_1d, bullish_bias)
    bearish_bias_aligned = align_htf_to_ltf(prices, df_1d, bearish_bias)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels
    donchian_upper, donchian_lower = calculate_donchian_channels(high, low, DONCHIAN_PERIOD)
    
    # ATR for stop loss
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    # Volume MA
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, ATR_PERIOD, VOLUME_MA_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or np.isnan(atr[i]) or np.isnan(volume_ma[i]):
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
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
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD)
        
        # Breakout conditions
        long_breakout = close[i] > donchian_upper[i]
        short_breakout = close[i] < donchian_lower[i]
        
        # Pivot bias from 1d
        bullish = bullish_bias_aligned[i]
        bearish = bearish_bias_aligned[i]
        
        # Generate signals
        if position == 0:
            # Long: breakout above Donchian upper + bullish bias + volume
            if long_breakout and bullish and volume_ok:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            # Short: breakout below Donchian lower + bearish bias + volume
            elif short_breakout and bearish and volume_ok:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long on opposite breakout or stop loss
            if short_breakout and bearish and volume_ok:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = SIGNAL_SIZE
        elif position == -1:
            # Exit short on opposite breakout or stop loss
            if long_breakout and bullish and volume_ok:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -SIGNAL_SIZE
    
    return signals