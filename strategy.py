#!/usr/bin/env python3
# Hypothesis: 6h Time-of-Day volatility breakout combined with 1-day ATR filter.
# During high volatility periods (UTC 12:00-20:00), price breaks often have follow-through.
# Uses ATR(14) from daily chart to set dynamic breakout thresholds, avoiding false breakouts in low volatility.
# Volatility filter ensures trades occur only when market has sufficient movement potential.
# Designed for 6h timeframe to target 50-150 total trades over 4 years (12-37/year).
# Works in both bull and bear markets by focusing on volatility expansion rather than direction.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Get daily data for ATR filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:  # Need enough for ATR calculation
        return np.zeros(n)
    
    # Calculate daily ATR (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Wilder's smoothing (equivalent to RMA)
    atr = np.zeros_like(tr)
    atr[0] = tr[0]
    for i in range(1, len(tr)):
        atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # Align ATR to 6h timeframe
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr)
    
    # Pre-calculate hour of day for each bar (vectorized)
    hours = pd.to_datetime(open_time).hour
    
    # Volatility breakout parameters
    breakout_mult = 0.5  # ATR multiplier for breakout threshold
    vol_threshold = 0.5  # Minimum ATR ratio to consider volatile enough
    
    # Calculate ATR ratio (current ATR / 20-period ATR average) for volatility regime
    atr_ma = pd.Series(atr_aligned).rolling(window=20, min_periods=20).mean().values
    atr_ratio = np.where(atr_ma > 0, atr_aligned / atr_ma, 0)
    
    signals = np.zeros(n)
    
    start_idx = max(30, 20)  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(atr_aligned[i]) or np.isnan(atr_ma[i]) or 
            np.isnan(atr_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Time filter: UTC 12:00-20:00 (high volatility period for crypto)
        hour = hours[i]
        in_volatile_hours = 12 <= hour <= 20
        
        # Volatility filter: only trade when ATR is above average
        volatile_enough = atr_ratio[i] > vol_threshold
        
        # Skip if not in trading hours or not volatile enough
        if not (in_volatile_hours and volatile_enough):
            signals[i] = 0.0
            continue
        
        # Calculate dynamic breakout levels based on previous bar
        if i > 0:
            prev_high = high[i-1]
            prev_low = low[i-1]
            prev_close = close[i-1]
            
            # Breakout thresholds
            upper_break = prev_high + breakout_mult * atr_aligned[i-1]
            lower_break = prev_low - breakout_mult * atr_aligned[i-1]
            
            # Breakout conditions
            breakout_up = close[i] > upper_break
            breakout_down = close[i] < lower_break
            
            # Additional confirmation: close must be beyond the midpoint of the range
            range_mid = (prev_high + prev_low) / 2
            confirmation_up = close[i] > range_mid
            confirmation_down = close[i] < range_mid
            
            # Entry signals
            if breakout_up and confirmation_up:
                signals[i] = 0.25
            elif breakout_down and confirmation_down:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_TimeOfDay_VolatilityBreakout_1dATR_Filter"
timeframe = "6h"
leverage = 1.0