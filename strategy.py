#!/usr/bin/env python3
"""
Experiment #10159: 6h Donchian Breakout + 12h Pivot + Volume Spike
Hypothesis: Donchian(20) breakouts confirmed by 12h pivot levels (R4/S4 for continuation, R3/S3 for reversal) 
with volume spike provide high-probability trades. Works in bull/bear by using pivot structure as dynamic 
support/resistance. Targets 75-150 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_ltf_to_htf

name = "exp_10159_6h_donchian_12h_pivot_volume_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
VOLUME_SPIKE_MULTIPLIER = 1.5
PIVOT_LOOKBACK = 10
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def calculate_donchian_channels(high, low, period):
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def calculate_pivot_points(high, low, close):
    """Calculate classic pivot points: P = (H+L+C)/3, R1=2P-L, S1=2P-H, R2=P+(H-L), S2=P-(H-L), etc."""
    pp = (high + low + close) / 3.0
    r1 = 2 * pp - low
    s1 = 2 * pp - high
    r2 = pp + (high - low)
    s2 = pp - (high - low)
    r3 = high + 2 * (pp - low)
    s3 = low - 2 * (high - pp)
    r4 = r3 + (high - low)
    s4 = s3 - (high - low)
    return pp, r1, s1, r2, s2, r3, s3, r4, s4

def calculate_atr(high, low, close, period):
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
    
    # Load 12h data ONCE for pivot points
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h pivot points
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    pp_12h, r1_12h, s1_12h, r2_12h, s2_12h, r3_12h, s3_12h, r4_12h, s4_12h = calculate_pivot_points(
        high_12h, low_12h, close_12h
    )
    
    # Align pivot levels to 6h timeframe
    r3_12h_aligned = align_htf_to_ltf(prices, df_12h, r3_12h)
    s3_12h_aligned = align_htf_to_ltf(prices, df_12h, s3_12h)
    r4_12h_aligned = align_htf_to_ltf(prices, df_12h, r4_12h)
    s4_12h_aligned = align_htf_to_ltf(prices, df_12h, s4_12h)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels
    donch_upper, donch_lower = calculate_donchian_channels(high, low, DONCHIAN_PERIOD)
    
    # Volume moving average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for risk management
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0
    entry_price = 0.0
    stop_price = 0.0
    
    start = max(DONCHIAN_PERIOD, 20) + 1
    
    for i in range(start, n):
        # Skip if pivot levels not available
        if np.isnan(r4_12h_aligned[i]) or np.isnan(s4_12h_aligned[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        # Check stoploss
        if position == 1:
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Volume spike confirmation
        volume_spike = volume[i] > (volume_ma[i] * VOLUME_SPIKE_MULTIPLIER) if not np.isnan(volume_ma[i]) else False
        
        # Breakout conditions
        bullish_breakout = close[i] > donch_upper[i] if not np.isnan(donch_upper[i]) else False
        bearish_breakout = close[i] < donch_lower[i] if not np.isnan(donch_lower[i]) else False
        
        # Pivot-based conditions
        # Strong breakout: price breaks beyond R4/S4 (continuation)
        strong_breakout_long = bullish_breakout and close[i] > r4_12h_aligned[i]
        strong_breakout_short = bearish_breakout and close[i] < s4_12h_aligned[i]
        
        # Fade at R3/S3: price rejects at R3/S3 (mean reversion)
        fade_long = close[i] < r3_12h_aligned[i] and close[i] > s3_12h_aligned[i] and bearish_breakout
        fade_short = close[i] > s3_12h_aligned[i] and close[i] < r3_12h_aligned[i] and bullish_breakout
        
        # Entry logic: strong breakouts with volume, or fades at R3/S3 with volume
        long_entry = (strong_breakout_long or fade_long) and volume_spike
        short_entry = (strong_breakout_short or fade_short) and volume_spike
        
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