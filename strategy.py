#!/usr/bin/env python3
"""
Experiment #8819: 6h Donchian breakout + 12h pivot direction + volume confirmation.
Hypothesis: 6h timeframe balances trade frequency with trend capture. Using 12h pivot points 
(R3/S3 for reversal, R4/S4 for breakout) provides institutional reference levels. Volume 
confirms institutional participation. Works in bull (breakouts) and bear (reversals at pivots).
Targets 50-150 total trades over 4 years (12-37/year) to avoid fee drag.
"""

from mtf_data import get_alt_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_8819_6h_donchian20_12h_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
PIVOT_LOOKBACK = 24  # 12h lookback for pivot calculation (4 periods of 6h)
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
    """Calculate pivot points: P = (H+L+C)/3, R1=2P-L, S1=2P-H, R2=P+(H-L), S2=P-(H-L),
       R3=H+2(P-L), S3=L-2(H-P), R4=R3+(H-L), S4=S3-(H-L)"""
    pivot = (high + low + close) / 3.0
    r1 = 2 * pivot - low
    s1 = 2 * pivot - high
    r2 = pivot + (high - low)
    s2 = pivot - (high - low)
    r3 = high + 2 * (pivot - low)
    s3 = low - 2 * (high - pivot)
    r4 = r3 + (high - low)
    s4 = s3 - (high - low)
    return pivot, r1, r2, r3, r4, s1, s2, s3, s4

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (12h for pivot points)
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h pivot points
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    _, _, _, r3_12h, r4_12h, _, _, s3_12h, s4_12h = calculate_pivot_points(high_12h, low_12h, close_12h)
    
    # Pivot signals: 1 = bullish bias (above R3), -1 = bearish bias (below S3)
    pivot_bias = np.where(close_12h > r3_12h, 1, 
                   np.where(close_12h < s3_12h, -1, 0))
    pivot_bias_aligned = align_htf_to_ltf(prices, df_12h, pivot_bias)
    
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
    start = max(DONCHIAN_PERIOD, PIVOT_LOOKBACK, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
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
        
        # Determine market bias from 12h pivot
        bull_bias = pivot_bias_aligned[i] == 1   # 12h close above R3
        bear_bias = pivot_bias_aligned[i] == -1  # 12h close below S3
        
        # Donchian breakout conditions
        long_breakout = close[i] > donchian_high[i-1]  # Break above previous period's high
        short_breakout = close[i] < donchian_low[i-1]  # Break below previous period's low
        
        # Volume confirmation
        volume_confirmed = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Entry conditions
        long_entry = bull_bias and long_breakout and volume_confirmed
        short_entry = bear_bias and short_breakout and volume_confirmed
        
        # Special case: fade at extreme pivots (R4/S4) in ranging markets
        # Only fade if no strong bias and at extreme levels
        no_bias = pivot_bias_aligned[i] == 0
        at_r4 = close[i] >= r4_12h[i] if not np.isnan(r4_12h[i]) else False
        at_s4 = close[i] <= s4_12h[i] if not np.isnan(s4_12h[i]) else False
        fade_long = no_bias and at_s4 and volume_confirmed  # bounce from S4
        fade_short = no_bias and at_r4 and volume_confirmed  # rejection at R4
        
        # Generate signals
        if position == 0:
            if long_entry or fade_long:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif short_entry or fade_short:
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