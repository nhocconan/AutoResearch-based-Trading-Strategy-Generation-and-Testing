#!/usr/bin/env python3
"""
exp_7439_6d_donchian20_12h_pivot_vol_v1
Hypothesis: 6h Donchian(20) breakout with 12h Camarilla pivot direction and volume confirmation.
Pivots from higher timeframe (12h) define institutional support/resistance. 
In bull/bear markets: fade at R3/S3, breakout continuation at R4/S4.
Volume filter ensures institutional participation. Targets 50-150 trades over 4 years.
"""

from mtf_data import get_taft_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_7439_6h_donchian20_12h_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
PIVOT_LOOKBACK = 1  # Use previous 12h bar for pivot calculation
VOL_MA_PERIOD = 20
VOL_BASE_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5
MAX_HOLD_BARS = 12  # 3 days max hold for 6h

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop - using 12h for pivot calculation
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h Camarilla pivots (using previous bar's OHLC)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Pivot = (H + L + C) / 3
    pivot = (high_12h + low_12h + close_12h) / 3.0
    # Range = H - L
    range_12h = high_12h - low_12h
    
    # Camarilla levels
    r3 = pivot + (range_12h * 1.1 / 2)
    s3 = pivot - (range_12h * 1.1 / 2)
    r4 = pivot + (range_12h * 1.1)
    s4 = pivot - (range_12h * 1.1)
    
    # Align pivots to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_12h, pivot)
    r3_aligned = align_htf_to_ltf(prices, df_12h, r3)
    s3_aligned = align_htf_to_ltf(prices, df_12h, s3)
    r4_aligned = align_htf_to_ltf(prices, df_12h, r4)
    s4_aligned = align_htf_to_ltf(prices, df_12h, s4)
    
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
    start = max(DONCHIAN_PERIOD, PIVOT_LOOKBACK, VOL_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        bars_since_entry += 1
        
        # Skip if HTF data not available
        if np.isnan(pivot_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]):
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
        
        # Determine pivot-based trading logic
        # Fade at R3/S3 (mean reversion), Breakout at R4/S4 (continuation)
        near_r3 = abs(close[i] - r3_aligned[i]) < (r4_aligned[i] - r3_aligned[i]) * 0.1
        near_s3 = abs(close[i] - s3_aligned[i]) < (s3_aligned[i] - s4_aligned[i]) * 0.1
        breakout_r4 = close[i] > r4_aligned[i]
        breakdown_s4 = close[i] < s4_aligned[i]
        
        # Fade signals at R3/S3 (expect reversal)
        fade_long = near_s3 and close[i] > s3_aligned[i] and vol_confirmed
        fade_short = near_r3 and close[i] < r3_aligned[i] and vol_confirmed
        
        # Breakout signals at R4/S4 (expect continuation)
        breakout_long = breakout_r4 and vol_confirmed
        breakout_short = breakdown_s4 and vol_confirmed
        
        # Donchian breakout confirmation
        donchian_long = close[i] > highest_high[i]
        donchian_short = close[i] < lowest_low[i]
        
        # Combine signals: require Donchian breakout in same direction
        long_signal = (fade_long or breakout_long) and donchian_long
        short_signal = (fade_short or breakout_short) and donchian_short
        
        # Enter new positions only if flat
        if position == 0:
            if long_signal:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                bars_since_entry = 0
            elif short_signal:
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