# 7767
#!/usr/bin/env python3
"""
Experiment #7767: 6-hour weekly pivot breakout with volume confirmation and ATR-based risk management.
Hypothesis: Price breaking beyond weekly R4/S4 levels with volume >1.5x 20-period MA captures breakout momentum, while fading at R3/S3 with reversal patterns captures mean reversion in ranging markets. Weekly pivot provides structural levels that work in both bull and bear markets.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_7767_6h_weekly_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
PIVOT_LOOKBACK = 5  # weeks for pivot calculation
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0
ATR_TARGET_MULTIPLIER = 3.0
BREAKOUT_BUFFER = 0.001  # 0.1% buffer to avoid false breakouts

def calculate_weekly_pivot(high, low, close):
    """Calculate weekly pivot points: R4, R3, S3, S4"""
    # Typical price for the week
    typical_price = (high + low + close) / 3.0
    # Weekly range
    weekly_range = high - low
    # Pivot point
    pivot = typical_price
    # Support and resistance levels
    s1 = (2 * pivot) - high
    r1 = (2 * pivot) - low
    s2 = pivot - (high - low)
    r2 = pivot + (high - low)
    s3 = low - 2.0 * (high - pivot)
    r3 = high + 2.0 * (pivot - low)
    s4 = s3 - (high - low)
    r4 = r3 + (high - low)
    return r4, r3, s3, s4

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop - weekly data for pivot points
    df_weekly = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot points
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    close_weekly = df_weekly['close'].values
    
    # Initialize arrays for pivot levels
    r4 = np.full_like(close_weekly, np.nan)
    r3 = np.full_like(close_weekly, np.nan)
    s3 = np.full_like(close_weekly, np.nan)
    s4 = np.full_like(close_weekly, np.nan)
    
    # Calculate pivots for each week
    for i in range(len(close_weekly)):
        r4[i], r3[i], s3[i], s4[i] = calculate_weekly_pivot(high_weekly[i], low_weekly[i], close_weekly[i])
    
    # Align weekly pivot levels to 6h timeframe (with shift(1) for completed weeks only)
    r4_aligned = align_htf_to_ltf(prices, df_weekly, r4)
    r3_aligned = align_htf_to_ltf(prices, df_weekly, r3)
    s3_aligned = align_htf_to_ltf(prices, df_weekly, s3)
    s4_aligned = align_htf_to_ltf(prices, df_weekly, s4)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume moving average
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR for risk management
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    target_price = 0.0
    
    # Start from warmup period
    start = max(VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if weekly pivot data not available
        if np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        # Check stoploss or target
        if position == 1:  # long position
            if close[i] <= stop_price or close[i] >= target_price:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # short position
            if close[i] >= stop_price or close[i] <= target_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Volume confirmation
        volume_confirmed = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Breakout conditions - price beyond weekly R4/S4 with buffer
        weekly_r4 = r4_aligned[i]
        weekly_r3 = r3_aligned[i]
        weekly_s3 = s3_aligned[i]
        weekly_s4 = s4_aligned[i]
        
        # Avoid division by zero or invalid levels
        if weekly_r4 <= 0 or weekly_s4 <= 0:
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        # Breakout signals
        long_breakout = (close[i] > weekly_r4 * (1 + BREAKOUT_BUFFER)) and volume_confirmed
        short_breakout = (close[i] < weekly_s4 * (1 - BREAKOUT_BUFFER)) and volume_confirmed
        
        # Mean reversion signals - price at R3/S3 with rejection
        # Look for rejection: price touches level but closes back inside
        long_reversion = (abs(close[i] - weekly_s3) < weekly_r4 * 0.002) and (close[i] > open[i]) and volume_confirmed
        short_reversion = (abs(close[i] - weekly_r3) < weekly_r4 * 0.002) and (close[i] < open[i]) and volume_confirmed
        
        # Entry conditions
        long_entry = long_breakout or long_reversion
        short_entry = short_breakout or short_reversion
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
                target_price = entry_price + (ATR_TARGET_MULTIPLIER * atr[i])
            elif short_entry:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
                target_price = entry_price - (ATR_TARGET_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals