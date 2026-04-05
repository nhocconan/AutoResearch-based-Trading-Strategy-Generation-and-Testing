#!/usr/bin/env python3
"""
Experiment #10155: 6h Mean Reversion at Weekly Pivots with Volume Exhaustion
Hypothesis: Price reversals at weekly Camarilla pivot levels (R3/S3) with volume exhaustion
provide high-probability mean reversion trades. Works in ranging markets and during
pullbacks in trending markets. Volume exhaustion (volume < 50% of average) confirms
lack of follow-through, increasing reversal probability. Target: 100-200 total trades
over 4 years (25-50/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_10155_6h_meanrev_weekly_pivots_volume_exhaustion_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
PIVOT_LOOKBACK = 5  # days for pivot calculation (use prior week)
VOLUME_EXHAUSTION_THRESHOLD = 0.5  # volume < 50% of average
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0
MIN_HOLD_BARS = 4  # minimum 4 bars (~1 day) before reversal allowed

def calculate_camarilla_pivots(high, low, close):
    """
    Calculate Camarilla pivot levels for the period
    Based on previous period's high, low, close
    R4 = close + ((high - low) * 1.500)
    R3 = close + ((high - low) * 1.250)
    R2 = close + ((high - low) * 1.166)
    R1 = close + ((high - low) * 1.083)
    PP = (high + low + close) / 3
    S1 = close - ((high - low) * 1.083)
    S2 = close - ((high - low) * 1.166)
    S3 = close - ((high - low) * 1.250)
    S4 = close - ((high - low) * 1.500)
    """
    pivot = (high + low + close) / 3.0
    range_val = high - low
    
    r4 = close + (range_val * 1.500)
    r3 = close + (range_val * 1.250)
    r2 = close + (range_val * 1.166)
    r1 = close + (range_val * 1.083)
    
    s1 = close - (range_val * 1.083)
    s2 = close - (range_val * 1.166)
    s3 = close - (range_val * 1.250)
    s4 = close - (range_val * 1.500)
    
    return r4, r3, r2, r1, pivot, s1, s2, s3, s4

def calculate_ema(close, period):
    """Calculate EMA"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

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
    if n < 50:
        return np.zeros(n)
    
    # Load daily data ONCE before loop for pivot calculation
    df_daily = get_htf_data(prices, '1d')
    
    # Calculate Camarilla pivots from previous day's HLC
    daily_high = df_daily['high'].values
    daily_low = df_daily['low'].values
    daily_close = df_daily['close'].values
    
    # Shift by 1 to use previous day's data (lookback)
    daily_high_shift = np.roll(daily_high, 1)
    daily_low_shift = np.roll(daily_low, 1)
    daily_close_shift = np.roll(daily_close, 1)
    # Set first value to NaN since we don't have previous day
    daily_high_shift[0] = np.nan
    daily_low_shift[0] = np.nan
    daily_close_shift[0] = np.nan
    
    r4, r3, r2, r1, pp, s1, s2, s3, s4 = calculate_camarilla_pivots(
        daily_high_shift, daily_low_shift, daily_close_shift
    )
    
    # Align pivot levels to 6h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_daily, r3)
    s3_aligned = align_htf_to_ltf(prices, df_daily, s3)
    r4_aligned = align_htf_to_ltf(prices, df_daily, r4)
    s4_aligned = align_htf_to_ltf(prices, df_daily, s4)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume moving average for exhaustion detection
    volume_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values  # 4 days
    
    # ATR for risk management
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = 25  # need volume MA and pivots
    
    for i in range(start, n):
        bars_since_entry += 1
        
        # Skip if pivots not available
        if np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]):
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
                bars_since_entry = 0
                continue
        elif position == -1:  # short position
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
                continue
        
        # Volume exhaustion: volume < 50% of average
        volume_exhausted = volume[i] < (volume_ma[i] * VOLUME_EXHAUSTION_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Proximity to pivot levels (within 0.5% ATR)
        proximity_threshold = atr[i] * 0.5
        near_r3 = abs(close[i] - r3_aligned[i]) <= proximity_threshold
        near_s3 = abs(close[i] - s3_aligned[i]) <= proximity_threshold
        
        # Rejection signals: price tests level but fails to break through
        # Long: price near S3 and closing above it (bounce)
        long_setup = near_s3 and close[i] > s3_aligned[i] and volume_exhausted
        # Short: price near R3 and closing below it (rejection)
        short_setup = near_r3 and close[i] < r3_aligned[i] and volume_exhausted
        
        # Additional filter: avoid trading against strong momentum
        # Use price change over last 2 bars to detect momentum
        if i >= 2:
            price_change_2bars = (close[i] - close[i-2]) / close[i-2]
            strong_up = price_change_2bars > 0.02  # >2% up
            strong_down = price_change_2bars < -0.02  # >2% down
        else:
            strong_up = False
            strong_down = False
        
        # Final entry conditions
        long_entry = long_setup and not strong_down and bars_since_entry >= MIN_HOLD_BARS
        short_entry = short_setup and not strong_up and bars_since_entry >= MIN_HOLD_BARS
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
                bars_since_entry = 0
            elif short_entry:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
                bars_since_entry = 0
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals