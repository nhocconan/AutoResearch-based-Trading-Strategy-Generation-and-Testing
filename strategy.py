#!/usr/bin/env python3
"""
exp_6738_1d_donchian20_1w_pivot_vol_v1
Hypothesis: Daily Donchian(20) breakout with weekly Camarilla pivot direction filter and volume confirmation.
In ranging markets: fade at R3/S3 levels. In trending markets: breakout continuation at R4/S4.
Weekly pivot provides structural support/resistance from higher timeframe. Volume confirms legitimacy.
Designed for 1d timeframe to capture longer-term swings with ~7-25 trades/year (30-100 total over 4 years).
Works in both bull and bear markets by adapting to weekly pivot context.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_6738_1d_donchian20_1w_pivot_vol_v1"
timeframe = "1d"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
VOL_MA_PERIOD = 20
VOL_BASE_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0
MAX_HOLD_BARS = 30  # ~6 weeks (1d bars)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop - using 1w for weekly pivot
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly Camarilla pivot levels
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Typical price for pivot calculation
    typical_1w = (high_1w + low_1w + close_1w) / 3
    pivot_1w = typical_1w
    range_1w = high_1w - low_1w
    
    # Camarilla levels
    r3_1w = pivot_1w + range_1w * 1.1 / 2
    s3_1w = pivot_1w - range_1w * 1.1 / 2
    r4_1w = pivot_1w + range_1w * 1.1
    s4_1w = pivot_1w - range_1w * 1.1
    
    # Align to LTF (1d)
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r3_1w_aligned = align_htf_to_ltf(prices, df_1w, r3_1w)
    s3_1w_aligned = align_htf_to_ltf(prices, df_1w, s3_1w)
    r4_1w_aligned = align_htf_to_ltf(prices, df_1w, r4_1w)
    s4_1w_aligned = align_htf_to_ltf(prices, df_1w, s4_1w)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels
    highest_high = pd.Series(high).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).max().values
    lowest_low = pd.Series(low).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).min().values
    
    # Volume MA for confirmation
    vol_ma = pd.Series(volume).rolling(window=VOL_MA_PERIOD, min_periods=VOL_MA_PERIOD).mean().values
    
    # ATR for stoploss
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, VOL_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        bars_since_entry += 1
        
        # Skip if HTF data not available
        if np.isnan(pivot_1w_aligned[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        # Check stoploss
        if position == 1:  # long position
            if close[i] <= entry_price - ATR_STOP_MULTIPLIER * atr[i]:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
                continue
        elif position == -1:  # short position
            if close[i] >= entry_price + ATR_STOP_MULTIPLIER * atr[i]:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
                continue
                
        # Time-based exit
        if position != 0 and bars_since_entry >= MAX_HOLD_BARS:
            signals[i] = 0.0
            position = 0
            bars_since_entry = 0
            continue
            
        # Volume confirmation
        vol_confirmed = volume[i] > vol_ma[i] * VOL_BASE_THRESHOLD if not np.isnan(vol_ma[i]) else False
        
        # Determine market regime based on weekly pivot
        # In ranging markets: price between S3 and R3 -> mean reversion at extremes
        # In trending markets: price outside S3/R3 -> breakout continuation
        in_range = (close[i] > s3_1w_aligned[i]) and (close[i] < r3_1w_aligned[i])
        
        # Mean reversion signals (range market)
        long_mean_revert = in_range and (close[i] <= s3_1w_aligned[i]) and vol_confirmed
        short_mean_revert = in_range and (close[i] >= r3_1w_aligned[i]) and vol_confirmed
        
        # Breakout continuation signals (trending market)
        long_breakout = (not in_range) and (close[i] > highest_high[i]) and vol_confirmed and (close[i] > r4_1w_aligned[i])
        short_breakout = (not in_range) and (close[i] < lowest_low[i]) and vol_confirmed and (close[i] < s4_1w_aligned[i])
        
        # Enter new positions only if flat
        if position == 0:
            if long_mean_revert or long_breakout:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                bars_since_entry = 0
            elif short_mean_revert or short_breakout:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                bars_since_entry = 0
            else:
                signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = position * SIGNAL_SIZE
    
    return signals