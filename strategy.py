#!/usr/bin/env python3
"""
exp_7319_6h_donchian20_12h_pivot_v1
Hypothesis: 6h Donchian(20) breakout with 12h Camarilla pivot levels for continuation/fade logic.
In trending markets: breakout continuation at R4/S4 levels with volume confirmation.
In ranging markets: mean reversion fade at R3/S3 levels with volume confirmation.
Uses 12h Camarilla pivots calculated from prior 12h bar's OHLC.
Designed for 6h timeframe to capture swings with ~12-37 trades/year (50-150 total over 4 years).
Works in both bull and bear markets by adapting to pivot-defined support/resistance structure.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_7319_6h_donchian20_12h_pivot_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
VOL_MA_PERIOD = 20
VOL_BASE_THRESHOLD = 1.8
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5
MAX_HOLD_BARS = 8  # ~48 hours

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop - using 12h for Camarilla pivots
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h Camarilla pivot levels (based on prior 12h bar's OHLC)
    # Camarilla formulas:
    # R4 = C + ((H-L) * 1.1/2)
    # R3 = C + ((H-L) * 1.1/4)
    # R2 = C + ((H-L) * 1.1/6)
    # R1 = C + ((H-L) * 1.1/12)
    # PP = (H+L+C)/3
    # S1 = C - ((H-L) * 1.1/12)
    # S2 = C - ((H-L) * 1.1/6)
    # S3 = C - ((H-L) * 1.1/4)
    # S4 = C - ((H-L) * 1.1/2)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate pivot levels using prior bar's values (shifted by 1 to avoid look-ahead)
    # We use shift(1) in align_htf_to_ltf, so we calculate on current bar but it gets shifted
    rng = high_12h - low_12h
    r4 = close_12h + (rng * 1.1 / 2)
    r3 = close_12h + (rng * 1.1 / 4)
    s3 = close_12h - (rng * 1.1 / 4)
    s4 = close_12h - (rng * 1.1 / 2)
    
    # Align to LTF (6h)
    r4_12h_aligned = align_htf_to_ltf(prices, df_12h, r4)
    r3_12h_aligned = align_htf_to_ltf(prices, df_12h, r3)
    s3_12h_aligned = align_htf_to_ltf(prices, df_12h, s3)
    s4_12h_aligned = align_htf_to_ltf(prices, df_12h, s4)
    
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
        if np.isnan(r4_12h_aligned[i]) or np.isnan(r3_12h_aligned[i]) or \
           np.isnan(s3_12h_aligned[i]) or np.isnan(s4_12h_aligned[i]):
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
        
        # Determine market structure based on price vs pivot levels
        # R4/S4 = strong breakout levels
        # R3/S3 = strong reversal levels
        at_r4 = close[i] >= r4_12h_aligned[i]
        at_r3 = close[i] >= r3_12h_aligned[i] and close[i] < r4_12h_aligned[i]
        at_s3 = close[i] <= s3_12h_aligned[i] and close[i] > s4_12h_aligned[i]
        at_s4 = close[i] <= s4_12h_aligned[i]
        
        # Breakout continuation at R4/S4 (trending market)
        breakout_long = at_r4 and (close[i] > highest_high[i]) and vol_confirmed
        breakout_short = at_s4 and (close[i] < lowest_low[i]) and vol_confirmed
        
        # Mean reversion fade at R3/S3 (ranging market)
        fade_long = at_s3 and (close[i] <= lowest_low[i]) and vol_confirmed
        fade_short = at_r3 and (close[i] >= highest_high[i]) and vol_confirmed
        
        # Enter new positions only if flat
        if position == 0:
            if breakout_long or fade_long:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                bars_since_entry = 0
            elif breakout_short or fade_short:
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