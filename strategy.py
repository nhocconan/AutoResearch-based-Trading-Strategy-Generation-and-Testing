#!/usr/bin/env python3
"""
Experiment #8867: 6h Donchian breakout + daily pivot + volume confirmation
Hypothesis: 6h timeframe reduces trade frequency vs lower timeframes while still capturing
meaningful trends. Daily pivot levels provide institutional reference points: price breaking
above R1/R2 with volume indicates bullish momentum, breaking below S1/S2 indicates bearish.
Volume confirmation ensures breakouts have institutional participation. Works in both bull
(breakouts continue) and bear (breakdowns continue) markets by trading with the 6h trend.
Targets 100-200 total trades over 4 years (25-50/year) to balance opportunity with cost control.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_8867_6h_donchian20_1d_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
PIVOT_LOOKBACK = 1  # Use previous day's pivot
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
    """Calculate standard pivot points: P, R1, R2, S1, S2"""
    pivot = (high + low + close) / 3.0
    r1 = 2 * pivot - low
    s1 = 2 * pivot - high
    r2 = pivot + (high - low)
    s2 = pivot - (high - low)
    return pivot, r1, r2, s1, s2

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop - daily data for pivot points
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily pivot points from previous day's OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivots for each day
    pivots = np.zeros_like(close_1d)
    r1_vals = np.zeros_like(close_1d)
    r2_vals = np.zeros_like(close_1d)
    s1_vals = np.zeros_like(close_1d)
    s2_vals = np.zeros_like(close_1d)
    
    for i in range(len(close_1d)):
        p, r1, r2, s1, s2 = calculate_pivot_points(high_1d[i], low_1d[i], close_1d[i])
        pivots[i] = p
        r1_vals[i] = r1
        r2_vals[i] = r2
        s1_vals[i] = s1
        s2_vals[i] = s2
    
    # Align daily pivots to 6h timeframe (shifted by 1 day for prior day's levels)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivots)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_vals)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2_vals)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_vals)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2_vals)
    
    # Calculate LTF indicators (6h)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
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
        # Skip if HTF data not available (first day)
        if np.isnan(pivot_aligned[i]):
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
        
        # Determine breakout conditions using prior day's pivot levels
        # Long: price breaks above R1 with volume
        long_breakout = close[i] > r1_aligned[i]
        # Short: price breaks below S1 with volume
        short_breakout = close[i] < s1_aligned[i]
        
        # Volume confirmation
        volume_confirmed = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Entry conditions
        long_entry = long_breakout and volume_confirmed
        short_entry = short_breakout and volume_confirmed
        
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
</x>