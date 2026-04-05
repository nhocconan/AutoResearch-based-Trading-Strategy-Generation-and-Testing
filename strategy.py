#!/usr/bin/env python3
"""
Experiment #8787: 6h Donchian breakout + daily pivot direction + volume confirmation + ATR stoploss.
Hypothesis: 6h timeframe balances trade frequency and signal quality. Daily pivot points (from prior day) provide
institutional reference levels - breaks above R1 or below S1 with volume confirmation indicate institutional
participation. This works in both bull and bear markets as it follows institutional flow rather than pure trend.
Targets 50-150 total trades over 4 years (12-37/year) to minimize fee impact while maintaining statistical validity.
"""

from mtf_data import get_alt_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_8787_6h_donchian20_1d_pivot_vol_v1"
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

def calculate_pivot(high, low, close):
    """Calculate classic pivot points: P = (H+L+C)/3, R1 = 2*P - L, S1 = 2*P - H"""
    pivot = (high + low + close) / 3.0
    r1 = 2 * pivot - low
    s1 = 2 * pivot - high
    return pivot, r1, s1

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop - daily data for pivots
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily pivot points from prior day's OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivots for each day
    pivot_vals = np.zeros_like(high_1d)
    r1_vals = np.zeros_like(high_1d)
    s1_vals = np.zeros_like(high_1d)
    
    for i in range(len(high_1d)):
        pivot_vals[i], r1_vals[i], s1_vals[i] = calculate_pivot(high_1d[i], low_1d[i], close_1d[i])
    
    # Shift by 1 to use prior day's pivot (avoid look-ahead)
    pivot_vals = np.roll(pivot_vals, 1)
    r1_vals = np.roll(r1_vals, 1)
    s1_vals = np.roll(s1_vals, 1)
    # First day will have rolled values from last - set to NaN
    pivot_vals[0] = np.nan
    r1_vals[0] = np.nan
    s1_vals[0] = np.nan
    
    # Align daily pivot data to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot_vals)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_vals)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_vals)
    
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
        if np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]):
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
        
        # Pivot-based signals: break above R1 = long, break below S1 = short
        long_breakout = close[i] > r1_aligned[i-1]  # Break above prior day's R1
        short_breakout = close[i] < s1_aligned[i-1]  # Break below prior day's S1
        
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