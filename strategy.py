#!/usr/bin/env python3
"""
Experiment #8687: 6h Donchian(20) breakout + 1d pivot direction + volume confirmation.
Hypothesis: 6h timeframe balances trade frequency and signal quality. Using daily pivot levels (R3/S3 for fade, R4/S4 for breakout) provides institutional reference points. Volume confirmation ensures breakouts have participation. Works in bull/bear by using pivot direction as filter.
Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag while maintaining statistical validity.
"""

from mtf_data import get_ftf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_8687_6h_donchian20_1d_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def calculate_pivot_points(high, low, close):
    """Calculate classic pivot points: P, R1, R2, R3, S1, S2, S3"""
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
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d pivot points
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Initialize pivot arrays
    r3 = np.full_like(close_1d, np.nan)
    s3 = np.full_like(close_1d, np.nan)
    r4 = np.full_like(close_1d, np.nan)
    s4 = np.full_like(close_1d, np.nan)
    
    # Calculate pivots for each day
    for i in range(len(close_1d)):
        _, _, _, r3_i, _, _, s3_i = calculate_pivot_points(high_1d[i], low_1d[i], close_1d[i])
        r3[i] = r3_i
        s3[i] = s3_i
        # R4/S4: extension of R3/S3 by same magnitude as R2-S2
        _, _, r2_i, _, _, s2_i, _ = calculate_pivot_points(high_1d[i], low_1d[i], close_1d[i])
        r4[i] = r3_i + (r2_i - s2_i)
        s4[i] = s3_i - (r2_i - s2_i)
    
    # Pivot bias: above R3 = bullish bias, below S3 = bearish bias
    pivot_bias = np.where(close_1d > r3, 1,  # bullish bias
                   np.where(close_1d < s3, -1, 0))  # bearish bias
    pivot_bias_aligned = align_htf_to_ltf(prices, df_1d, pivot_bias)
    
    # Calculate LTF indicators (6h)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels
    donchian_high = pd.Series(high).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).max().values
    donchian_low = pd.Series(low).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).min().values
    
    # Volume moving average
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR for risk management
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(pivot_bias_aligned[i]):
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
        
        # Determine market bias from 1d pivots
        bull_bias = pivot_bias_aligned[i] == 1   # 1d close above R3
        bear_bias = pivot_bias_aligned[i] == -1  # 1d close below S3
        
        # Donchian breakout conditions
        long_breakout = close[i] > donchian_high[i-1]  # Break above previous period's high
        short_breakout = close[i] < donchian_low[i-1]  # Break below previous period's low
        
        # Volume confirmation
        volume_confirmed = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Entry conditions
        long_entry = bull_bias and long_breakout and volume_confirmed
        short_entry = bear_bias and short_breakout and volume_confirmed
        
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