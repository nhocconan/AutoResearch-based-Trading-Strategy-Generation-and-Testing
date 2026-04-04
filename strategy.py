#!/usr/bin/env python3
"""
exp_6563_4h_donchian20_1d_ema_vol_v1
Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation.
Uses discrete sizing (0.30) and adaptive volume threshold to target 75-200 total trades over 4 years.
Works in both bull/bear markets via 1d EMA50 trend filter.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_6563_4h_donchian20_1d_ema_vol_v1"
timeframe = "4h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
EMA_PERIOD = 50         # 1d EMA50 for long-term trend filter
VOL_MA_PERIOD = 20
VOL_BASE_THRESHOLD = 1.8  # Base volume threshold
VOL_DYNAMIC_FACTOR = 0.5  # Factor for adaptive threshold based on volatility
SIGNAL_SIZE = 0.30      # 30% position size

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop - using 1d for EMA50
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA50
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=EMA_PERIOD, adjust=False).mean().values
    
    # Align to LTF (4h) with shift(1) for completed bars only
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).max().values
    donchian_low = pd.Series(low).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).min().values
    
    # Volume MA for confirmation
    vol_ma = pd.Series(volume).rolling(window=VOL_MA_PERIOD, min_periods=VOL_MA_PERIOD).mean().values
    
    # Volatility measure for adaptive volume threshold (ATR-like using high-low)
    hl_range = high - low
    vol_measure = pd.Series(hl_range).rolling(window=20, min_periods=1).mean().values
    vol_measure_ma = pd.Series(vol_measure).rolling(window=20, min_periods=20).mean().values
    
    # Adaptive volume threshold: higher in volatile periods, lower in calm periods
    vol_ratio = np.where(vol_measure_ma > 0, vol_measure / vol_measure_ma, 1.0)
    vol_threshold = VOL_BASE_THRESHOLD * (1.0 + VOL_DYNAMIC_FACTOR * (vol_ratio - 1.0))
    vol_threshold = np.clip(vol_threshold, 1.5, 2.5)  # Keep within reasonable bounds
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, VOL_MA_PERIOD) + 1
    
    for i in range(start, n):
        bars_since_entry += 1
        
        # Skip if HTF data not available
        if np.isnan(ema_1d_aligned[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        # Long conditions: price > 1d EMA50 (bullish bias) + breaks above Donchian HIGH + volume spike
        long_bias = close[i] > ema_1d_aligned[i]  # price above 1d EMA50 (bullish)
        long_breakout = close[i] > donchian_high[i-1]  # break above previous period's high
        long_volume = volume[i] > vol_ma[i] * vol_threshold[i] if not np.isnan(vol_ma[i]) and not np.isnan(vol_threshold[i]) else False
        
        # Short conditions: price < 1d EMA50 (bearish bias) + breaks below Donchian LOW + volume spike
        short_bias = close[i] < ema_1d_aligned[i]  # price below 1d EMA50 (bearish)
        short_breakout = close[i] < donchian_low[i-1]  # break below previous period's low
        short_volume = volume[i] > vol_ma[i] * vol_threshold[i] if not np.isnan(vol_ma[i]) and not np.isnan(vol_threshold[i]) else False
        
        # Exit conditions: EMA reversal OR time-based exit (max 10 bars) OR Donchian midpoint reversal
        if position == 1:  # long position
            # Exit if price drops back below EMA50 (trend change)
            exit_long = close[i] < ema_1d_aligned[i]
            # Or if price drops below Donchian midpoint
            exit_long = exit_long or close[i] < (donchian_high[i-1] + donchian_low[i-1]) / 2
            # Time-based exit: prevent overstaying
            exit_long = exit_long or bars_since_entry >= 10
            if exit_long:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
                continue
        elif position == -1:  # short position
            # Exit if price rises back above EMA50 (trend change)
            exit_short = close[i] > ema_1d_aligned[i]
            # Or if price rises above Donchian midpoint
            exit_short = exit_short or close[i] > (donchian_high[i-1] + donchian_low[i-1]) / 2
            # Time-based exit: prevent overstaying
            exit_short = exit_short or bars_since_entry >= 10
            if exit_short:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
                continue
        
        # Enter new positions only if flat
        if position == 0:
            if long_bias and long_breakout and long_volume:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                bars_since_entry = 0
            elif short_bias and short_breakout and short_volume:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                bars_since_entry = 0
            else:
                signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = position * SIGNAL_SIZE
    
    return signals