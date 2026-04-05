#!/usr/bin/env python3
"""
Experiment #9415: 6h Donchian breakout + weekly pivot direction + volume confirmation.
Hypothesis: Donchian breakouts capture trends; weekly pivot provides directional bias (price above/below weekly pivot); volume confirms institutional participation.
Designed for both bull and bear markets: in bull, take long breakouts above weekly pivot; in bear, take short breakdowns below weekly pivot.
Targets 100-200 total trades over 4 years (25-50/year) to balance opportunity and cost.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_9415_6h_donchian20_weekly_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
PIVOT_LOOKBACK = 1  # Use previous week's pivot
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.8
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.2

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def calculate_weekly_pivot(high, low, close):
    """Calculate weekly pivot point: (H + L + C) / 3"""
    return (high + low + close) / 3.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load weekly data ONCE before loop
    df_weekly = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot from previous week's data
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    weekly_pivot = calculate_weekly_pivot(weekly_high, weekly_low, weekly_close)
    
    # Price relative to weekly pivot: above = bullish bias, below = bearish bias
    price_vs_pivot = np.where(weekly_close > weekly_pivot, 1, 
                     np.where(weekly_close < weekly_pivot, -1, 0))  # 1=bullish, -1=bearish, 0=at pivot
    price_vs_pivot_aligned = align_htf_to_ltf(prices, df_weekly, price_vs_pivot)
    
    # Calculate 6h indicators
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
        # Skip if weekly data not available
        if np.isnan(price_vs_pivot_aligned[i]):
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
        
        # Determine market bias from weekly pivot
        bull_bias = price_vs_pivot_aligned[i] == 1   # Weekly close above pivot
        bear_bias = price_vs_pivot_aligned[i] == -1  # Weekly close below pivot
        
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
</file>