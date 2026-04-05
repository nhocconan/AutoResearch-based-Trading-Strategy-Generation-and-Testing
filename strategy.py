#!/usr/bin/env python3
"""
Experiment #9991: 6h Camarilla Pivot Reversal + Volume Spike
Hypothesis: In ranging markets, price often reverses at Camarilla pivot levels (R3/S3, R4/S4).
In trending markets, breakouts beyond R4/S4 with volume continuation provide trend-following entries.
Works in both bull and bear markets by adapting to regime via price position relative to daily pivot.
Target: 75-150 total trades over 4 years (19-38/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_9991_6h_camarilla_pivot_reversal_volume_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
CAMARILLA_MULT = 1.1  # Standard Camarilla uses 1.1
VOLUME_SPIKE_MULTIPLIER = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0
PIVOT_LOOKBACK = 5  # Require pivot to be at least this old

def calculate_camarilla_pivots(high, low, close):
    """Calculate Camarilla pivot levels for the day"""
    # Typical price
    pp = (high + low + close) / 3.0
    range_ = high - low
    
    # Camarilla levels
    r4 = pp + (range_ * 1.1 * 2)
    r3 = pp + (range_ * 1.1)
    s3 = pp - (range_ * 1.1)
    s4 = pp - (range_ * 1.1 * 2)
    
    return pp, r3, r4, s3, s4

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
    
    # Load daily data ONCE before loop for pivot levels
    df_daily = get_htf_data(prices, '1d')
    
    # Calculate Camarilla pivots from daily data
    daily_high = df_daily['high'].values
    daily_low = df_daily['low'].values
    daily_close = df_daily['close'].values
    
    pp, r3, r4, s3, s4 = calculate_camarilla_pivots(daily_high, daily_low, daily_close)
    
    # Align daily pivot levels to 6h timeframe (shifted by 1 day for no look-ahead)
    pp_aligned = align_htf_to_ltf(prices, df_daily, pp)
    r3_aligned = align_htf_to_ltf(prices, df_daily, r3)
    r4_aligned = align_htf_to_ltf(prices, df_daily, r4)
    s3_aligned = align_htf_to_ltf(prices, df_daily, s3)
    s4_aligned = align_htf_to_ltf(prices, df_daily, s4)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume moving average for spike detection
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for risk management
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(20, PIVOT_LOOKBACK) + 1
    
    for i in range(start, n):
        # Skip if daily pivots not available
        if np.isnan(pp_aligned[i]):
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
        
        # Price position relative to pivots
        at_r3 = abs(close[i] - r3_aligned[i]) < (r4_aligned[i] - r3_aligned[i]) * 0.1  # Within 10% of R3
        at_s3 = abs(close[i] - s3_aligned[i]) < (s3_aligned[i] - s4_aligned[i]) * 0.1  # Within 10% of S3
        above_r4 = close[i] > r4_aligned[i]
        below_s4 = close[i] < s4_aligned[i]
        
        # Entry conditions
        long_entry = False
        short_entry = False
        
        # Reversal at R3/S3 with volume spike (mean reversion in range)
        if volume_spike:
            if at_r3 and close[i] < r3_aligned[i]:  # Rejection at R3
                short_entry = True
            if at_s3 and close[i] > s3_aligned[i]:  # Bounce at S3
                long_entry = True
        
        # Breakout beyond R4/S4 with volume (trend continuation)
        if volume_spike:
            if above_r4:
                long_entry = True  # Break above R4
            if below_s4:
                short_entry = True  # Break below S4
        
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