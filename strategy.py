#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_13947_6d_donchian20_1w_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

# Hypothesis: 6h Donchian(20) breakout with weekly pivot direction filter and volume confirmation (1.5x)
# Weekly pivots provide strong institutional support/resistance levels
# In bull markets: buy breakouts above weekly R1 with volume
# In bear markets: sell breakdowns below weekly S1 with volume
# Uses weekly timeframe for pivot calculation to avoid noise and capture major levels
# Target: 50-150 total trades over 4 years via strict volume filter and pivot alignment

def calculate_pivot_points(high, low, close):
    """Calculate classic pivot points: P, R1, R2, R3, S1, S2, S3"""
    p = (high + low + close) / 3.0
    r1 = 2 * p - low
    s1 = 2 * p - high
    r2 = p + (high - low)
    s2 = p - (high - low)
    r3 = high + 2 * (p - low)
    s3 = low - 2 * (high - p)
    return p, r1, r2, r3, s1, s2, s3

def calculate_donchian(high, low, period):
    """Calculate Donchian upper and lower bands"""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data for pivot points ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot points
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate all pivot levels
    p_1w, r1_1w, r2_1w, r3_1w, s1_1w, s2_1w, s3_1w = calculate_pivot_points(high_1w, low_1w, close_1w)
    
    # Align pivot levels to 6h timeframe (shifted by 1 week for completed bars only)
    p_1w_aligned = align_htf_to_ltf(prices, df_1w, p_1w)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    
    # 6h data for Donchian, ATR, and volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    
    # ATR for stop loss
    atr = calculate_atr(high, low, close, 14)
    
    # Volume confirmation
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(20, 14, 20) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(p_1w_aligned[i]) or np.isnan(r1_1w_aligned[i]) or np.isnan(s1_1w_aligned[i]) or
            np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check stops
        if position == 1:  # long position
            # Check stop loss
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # short position
            # Check stop loss
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Volume confirmation - moderate threshold to balance signal quality and frequency
        volume_ok = volume[i] > (volume_ma[i] * 1.5)
        
        # Breakout signals
        breakout_up = close[i] > donchian_upper[i-1]  # break above previous upper band
        breakout_down = close[i] < donchian_lower[i-1]  # break below previous lower band
        
        # Weekly pivot filters
        # Long: price above weekly R1 (bullish bias)
        # Short: price below weekly S1 (bearish bias)
        bias_up = close[i] > r1_1w_aligned[i]
        bias_down = close[i] < s1_1w_aligned[i]
        
        # Entry signals
        long_signal = volume_ok and breakout_up and bias_up
        short_signal = volume_ok and breakout_down and bias_down
        
        # Generate signals
        if position == 0:
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (2.0 * atr[i])
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (2.0 * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long on Donchian breakdown or price falls below weekly pivot
            if close[i] < donchian_lower[i] or close[i] < p_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short on Donchian breakout or price rises above weekly pivot
            if close[i] > donchian_upper[i] or close[i] > p_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals