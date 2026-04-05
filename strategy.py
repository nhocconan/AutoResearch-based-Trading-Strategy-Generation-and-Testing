#!/usr/bin/env python3
"""
Experiment #11735: 6h Donchian Breakout with Weekly Pivot and Volume Confirmation
Hypothesis: Weekly pivot levels provide strong institutional support/resistance. 
Breakouts above weekly R1 or below weekly S1 with volume confirmation capture
trend continuation, while rejections at these levels offer mean reversion opportunities.
The weekly filter reduces false breakouts in choppy markets, improving win rate.
Designed for 6h timeframe to balance trade frequency (~20-40/year) and capture
multi-day trends in BTC/ETH/SOL across bull/bear regimes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_11735_6h_donchian20_weekly_pivot_vol"
timeframe = "6h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
PIVOT_LOOKBACK = 1  # Use previous week's pivot
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def calculate_pivot_points(high, low, close):
    """Calculate weekly pivot points (standard formula)"""
    pivot = (high + low + close) / 3.0
    r1 = 2 * pivot - low
    s1 = 2 * pivot - high
    r2 = pivot + (high - low)
    s2 = pivot - (high - low)
    r3 = high + 2 * (pivot - low)
    s3 = low - 2 * (high - pivot)
    return pivot, r1, r2, r3, s1, s2, s3

def calculate_donchian_channels(high, low, period):
    """Calculate Donchian channels"""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def calculate_ema(close, period):
    """Calculate EMA"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

def calculate_atr(high, low, close, period):
    """Calculate ATR"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data ONCE before loop
    df_weekly = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot points (using weekly OHLC)
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    # Calculate pivots for each week
    pivots = np.full_like(weekly_high, np.nan)
    r1 = np.full_like(weekly_high, np.nan)
    s1 = np.full_like(weekly_high, np.nan)
    
    for i in range(len(weekly_high)):
        if not (np.isnan(weekly_high[i]) or np.isnan(weekly_low[i]) or np.isnan(weekly_close[i])):
            p, r1_val, _, _, s1_val, _, _ = calculate_pivot_points(
                weekly_high[i], weekly_low[i], weekly_close[i]
            )
            pivots[i] = p
            r1[i] = r1_val
            s1[i] = s1_val
    
    # Align weekly data to 6h timeframe (shifted by 1 week for no look-ahead)
    pivots_aligned = align_htf_to_ltf(prices, df_weekly, pivots)
    r1_aligned = align_htf_to_ltf(prices, df_weekly, r1)
    s1_aligned = align_htf_to_ltf(prices, df_weekly, s1)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    donchian_upper, donchian_lower = calculate_donchian_channels(high, low, DONCHIAN_PERIOD)
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, VOLUME_MA_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if weekly data not available
        if np.isnan(pivots_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]):
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
        
        # Donchian breakout conditions (using previous bar)
        breakout_up = high[i] > donchian_upper[i-1] if i > 0 and not np.isnan(donchian_upper[i-1]) else False
        breakout_down = low[i] < donchian_lower[i-1] if i > 0 and not np.isnan(donchian_lower[i-1]) else False
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Weekly pivot conditions
        above_weekly_r1 = close[i] > r1_aligned[i]
        below_weekly_s1 = close[i] < s1_aligned[i]
        between_pivots = (close[i] >= s1_aligned[i]) and (close[i] <= r1_aligned[i])
        
        # Entry conditions: 
        # Long: breakout above Donchian AND above weekly R1 (bullish breakout) with volume
        # Short: breakdown below Donchian AND below weekly S1 (bearish breakdown) with volume
        long_entry = breakout_up and above_weekly_r1 and volume_ok
        short_entry = breakout_down and below_weekly_s1 and volume_ok
        
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
</|endoftext|>