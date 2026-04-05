#!/usr/bin/env python3
"""
Experiment #10555: 6h Donchian Breakout + Weekly Pivot Direction + Volume Spike
Hypothesis: 6h Donchian(20) breakouts in the direction of weekly Camarilla pivot levels (above H4 = long, below L4 = short)
with volume confirmation provide high-probability trend continuation. Works in bull markets (breakouts above weekly H4)
and bear markets (breakdowns below weekly L4). Volume filters reduce false breakouts.
Target: 75-200 total trades over 4 years (19-50/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_10555_6h_donchian_breakout_weekly_pivot_volume_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
VOLUME_SPIKE_MULTIPLIER = 1.8
PIVOT_PERIOD = 1  # weekly pivot from previous week
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def calculate_camarilla_pivot(high, low, close):
    """Calculate Camarilla pivot levels for the period"""
    # Typical price
    typical = (high + low + close) / 3
    # Range
    range_val = high - low
    # Camarilla levels
    h4 = close + (range_val * 1.1 / 2)
    l4 = close - (range_val * 1.1 / 2)
    h3 = close + (range_val * 1.1 / 4)
    l3 = close - (range_val * 1.1 / 4)
    h2 = close + (range_val * 1.1 / 6)
    l2 = close - (range_val * 1.1 / 6)
    h1 = close + (range_val * 1.1 / 12)
    l1 = close - (range_val * 1.1 / 12)
    return h4, l4, h3, l3, h2, l2, h1, l1

def calculate_donchian_channels(high, low, period):
    """Calculate Donchian channels"""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    # First ATR is simple average
    atr = np.zeros_like(tr)
    atr[period-1] = np.mean(tr[:period])
    # Wilder's smoothing: ATR[t] = (ATR[t-1] * (period-1) + TR[t]) / period
    for i in range(period, len(tr)):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data ONCE before loop for pivot levels
    df_weekly = get_htf_data(prices, '1w')
    
    # Calculate weekly Camarilla pivot levels
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    # Calculate pivots for each weekly bar
    h4_weekly = np.full(len(weekly_close), np.nan)
    l4_weekly = np.full(len(weekly_close), np.nan)
    
    for i in range(len(weekly_close)):
        if not (np.isnan(weekly_high[i]) or np.isnan(weekly_low[i]) or np.isnan(weekly_close[i])):
            h4, l4, _, _, _, _, _, _ = calculate_camarilla_pivot(
                weekly_high[i], weekly_low[i], weekly_close[i]
            )
            h4_weekly[i] = h4
            l4_weekly[i] = l4
    
    # Align weekly pivot levels to 6h timeframe
    h4_weekly_aligned = align_htf_to_ltf(prices, df_weekly, h4_weekly)
    l4_weekly_aligned = align_htf_to_ltf(prices, df_weekly, l4_weekly)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels
    donch_upper, donch_lower = calculate_donchian_channels(high, low, DONCHIAN_PERIOD)
    
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
        # Skip if weekly pivots not available
        if np.isnan(h4_weekly_aligned[i]) or np.isnan(l4_weekly_aligned[i]):
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
        
        # Pivot filter: price above/below weekly H4/L4
        above_weekly_h4 = close[i] > h4_weekly_aligned[i]
        below_weekly_l4 = close[i] < l4_weekly_aligned[i]
        
        # Breakout conditions
        bullish_breakout = close[i] > donch_upper[i] if not np.isnan(donch_upper[i]) else False
        bearish_breakout = close[i] < donch_lower[i] if not np.isnan(donch_lower[i]) else False
        
        # Entry conditions: breakout in direction of weekly pivot with volume
        long_entry = bullish_breakout and above_weekly_h4 and volume_spike
        short_entry = bearish_breakout and below_weekly_l4 and volume_spike
        
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