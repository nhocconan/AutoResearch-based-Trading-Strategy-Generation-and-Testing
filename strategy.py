#!/usr/bin/env python3
"""
Experiment #8447: 6h Donchian breakout + daily pivot direction + volume confirmation
Hypothesis: Daily pivot levels provide institutional reference points (R3/S3 for reversal, R4/S4 for breakout).
Combined with 6h Donchian breakouts and volume confirmation, this captures institutional flow at key levels.
Works in bull/bear: Pivots adapt to price levels, Donchian captures breakouts, volume filters false signals.
Target: 50-150 trades over 4 years (12-37/year) to balance frequency and statistical validity.
"""

from mtf_data import get_athf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_8447_6h_donchian20_1d_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
PIVOT_LOOKBACK = 1
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def calculate_pivot_points(high, low, close):
    """Calculate standard pivot points: P = (H+L+C)/3, R1=2P-L, S1=2P-H, etc."""
    p = (high + low + close) / 3.0
    r1 = 2 * p - low
    s1 = 2 * p - high
    r2 = p + (high - low)
    s2 = p - (high - low)
    r3 = high + 2 * (p - low)
    s3 = low - 2 * (high - p)
    r4 = r3 + (high - low)
    s4 = s3 - (high - low)
    return p, r1, s1, r2, s2, r3, s3, r4, s4

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily pivot points
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Initialize pivot arrays
    r3_1d = np.full_like(high_1d, np.nan)
    s3_1d = np.full_like(high_1d, np.nan)
    r4_1d = np.full_like(high_1d, np.nan)
    s4_1d = np.full_like(high_1d, np.nan)
    
    # Calculate pivots for each day
    for i in range(len(high_1d)):
        _, _, _, _, _, r3, s3, r4, s4 = calculate_pivot_points(high_1d[i], low_1d[i], close_1d[i])
        r3_1d[i] = r3
        s3_1d[i] = s3
        r4_1d[i] = r4
        s4_1d[i] = s4
    
    # Pivot signals: 1 = bullish bias (above R3), -1 = bearish bias (below S3), 0 = neutral
    pivot_signal = np.where(close_1d > r3_1d, 1,
                   np.where(close_1d < s3_1d, -1, 0))
    pivot_signal_aligned = align_htf_to_ltf(prices, df_1d, pivot_signal)
    
    # Breakout signals: 1 = break above R4, -1 = break below S4
    breakout_signal = np.where(close_1d > r4_1d, 1,
                    np.where(close_1d < s4_1d, -1, 0))
    breakout_signal_aligned = align_htf_to_ltf(prices, df_1d, breakout_signal)
    
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
        if np.isnan(pivot_signal_aligned[i]) or np.isnan(breakout_signal_aligned[i]):
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
        
        # Determine market context from daily pivots
        pivot_bull = pivot_signal_aligned[i] == 1   # Daily close above R3
        pivot_bear = pivot_signal_aligned[i] == -1  # Daily close below S3
        breakout_up = breakout_signal_aligned[i] == 1   # Daily break above R4
        breakout_down = breakout_signal_aligned[i] == -1  # Daily break below S4
        
        # 6h Donchian breakout conditions
        donchian_breakout_up = close[i] > donchian_high[i-1]  # Break above 6h Donchian high
        donchian_breakout_down = close[i] < donchian_low[i-1]  # Break below 6h Donchian low
        
        # Volume confirmation
        volume_confirmed = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Entry logic:
        # 1. In bullish daily context (above R3): look for long on 6h Donchian breakout OR daily breakout up
        # 2. In bearish daily context (below S3): look for short on 6h Donchian breakout OR daily breakout down
        # 3. Require volume confirmation for all entries
        
        long_entry = False
        short_entry = False
        
        if pivot_bull or breakout_up:
            # Bullish bias: look for longs
            if (donchian_breakout_up or breakout_up) and volume_confirmed:
                long_entry = True
        
        if pivot_bear or breakout_down:
            # Bearish bias: look for shorts
            if (donchian_breakout_down or breakout_down) and volume_confirmed:
                short_entry = True
        
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