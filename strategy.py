#!/usr/bin/env python3
"""
exp_7355_6h_donchian20_1w_pivot_vol_v1
Hypothesis: 6h Donchian(20) breakout with 1w Camarilla pivot direction filter and volume confirmation.
Uses weekly pivot levels (R3/S3 for fade, R4/S4 for breakout) to capture institutional levels.
Volume spike confirms breakout legitimacy. Discrete sizing (0.0, ±0.25) minimizes fee churn.
Designed for 6h timeframe targeting 50-150 total trades over 4 years.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_7355_6h_donchian20_1w_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
VOL_MA_PERIOD = 20
VOL_BASE_THRESHOLD = 2.0
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5
MAX_HOLD_BARS = 8  # ~2 days

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop - using 1w for Camarilla pivots
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w OHLC for Camarilla pivots
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Camarilla levels: based on previous week's range
    # R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low)
    # S3 = close - 1.1*(high-low), S4 = close - 1.5*(high-low)
    pivot_range = high_1w - low_1w
    camarilla_r4 = close_1w + 1.5 * pivot_range
    camarilla_r3 = close_1w + 1.1 * pivot_range
    camarilla_s3 = close_1w - 1.1 * pivot_range
    camarilla_s4 = close_1w - 1.5 * pivot_range
    
    # Align to LTF (6h)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r4)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s3)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s4)
    
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
        if np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_s4_aligned[i]):
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
        
        # Determine Camarilla zones
        above_r4 = close[i] > camarilla_r4_aligned[i]
        below_s4 = close[i] < camarilla_s4_aligned[i]
        between_r3_r4 = camarilla_r3_aligned[i] < close[i] < camarilla_r4_aligned[i]
        between_s3_s4 = camarilla_s4_aligned[i] < close[i] < camarilla_s3_aligned[i]
        
        # Breakout continuation at R4/S4 with volume
        breakout_long = above_r4 and vol_confirmed
        breakout_short = below_s4 and vol_confirmed
        
        # Fade at R3/S3 (mean reversion from extreme levels)
        fade_long = between_s3_s4 and not vol_confirmed and close[i] > camarilla_s3_aligned[i]
        fade_short = between_r3_r4 and not vol_confirmed and close[i] < camarilla_r3_aligned[i]
        
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