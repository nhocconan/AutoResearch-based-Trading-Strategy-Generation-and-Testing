#!/usr/bin/env python3
"""
exp_6967_6h_camarilla1d_pivot_v1
Hypothesis: 6h Camarilla pivot levels from 1d timeframe with volume confirmation.
Fade at R3/S3 levels in ranging markets (CHOP > 61.8), breakout continuation at R4/S4 in trending markets (CHOP < 38.2).
Uses 1d HTF for pivot calculation and regime detection to avoid false signals in choppy conditions.
Designed for 6h timeframe to capture meaningful swings with ~12-37 trades/year (50-150 total over 4 years).
Works in both bull and bear markets by adapting to regime via Choppiness Index filter.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_6967_6h_camarilla1d_pivot_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
VOL_MA_PERIOD = 20
VOL_BASE_THRESHOLD = 1.8
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5
MAX_HOLD_BARS = 24  # ~6 days (6h bars)
CHOPPINESS_PERIOD = 14
CHOPPINESS_RANGE_THRESHOLD = 61.8
CHOPPINESS_TREND_THRESHOLD = 38.2

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop - using 1d for Camarilla pivots and regime
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d OHLC for Camarilla pivots
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for 1d
    # R4 = C + ((H-L) * 1.1/2), R3 = C + ((H-L) * 1.1/4), etc.
    camarilla_r4 = close_1d + (high_1d - low_1d) * 1.1 / 2
    camarilla_r3 = close_1d + (high_1d - low_1d) * 1.1 / 4
    camarilla_s3 = close_1d - (high_1d - low_1d) * 1.1 / 4
    camarilla_s4 = close_1d - (high_1d - low_1d) * 1.1 / 2
    
    # Calculate Choppiness Index for regime detection (1d)
    tr1_1d = pd.Series(high_1d - low_1d)
    tr2_1d = pd.Series(np.abs(high_1d - np.roll(close_1d, 1)))
    tr3_1d = pd.Series(np.abs(low_1d - np.roll(close_1d, 1)))
    tr_1d = pd.concat([tr1_1d, tr2_1d, tr3_1d], axis=1).max(axis=1)
    atr_1d = tr_1d.ewm(span=CHOPPINESS_PERIOD, adjust=False, min_periods=CHOPPINESS_PERIOD).mean().values
    chop_denominator = atr_1d * CHOPPINESS_PERIOD
    chop_numerator = pd.Series(tr_1d).rolling(window=CHOPPINESS_PERIOD, min_periods=CHOPPINESS_PERIOD).sum().values
    chopiness = 100 * np.log10(chop_numerator / chop_denominator) / np.log10(CHOPPINESS_PERIOD)
    
    # Align HTF indicators to LTF (6h)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    chopiness_aligned = align_htf_to_ltf(prices, df_1d, chopiness)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
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
    start = max(VOL_MA_PERIOD, ATR_PERIOD, CHOPPINESS_PERIOD) + 1
    
    for i in range(start, n):
        bars_since_entry += 1
        
        # Skip if HTF data not available
        if np.isnan(camarilla_r3_aligned[i]) or np.isnan(chopiness_aligned[i]):
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
        vol_ma = pd.Series(volume).rolling(window=VOL_MA_PERIOD, min_periods=VOL_MA_PERIOD).mean().values
        vol_confirmed = volume[i] > vol_ma[i] * VOL_BASE_THRESHOLD if not np.isnan(vol_ma[i]) else False
        
        # Regime detection from Choppiness Index
        is_ranging = chopiness_aligned[i] > CHOPPINESS_RANGE_THRESHOLD
        is_trending = chopiness_aligned[i] < CHOPPINESS_TREND_THRESHOLD
        
        # Trading logic based on regime
        long_signal = False
        short_signal = False
        
        if is_ranging:
            # In ranging markets: fade at R3/S3 (mean reversion)
            long_signal = close[i] <= camarilla_s3_aligned[i] and vol_confirmed
            short_signal = close[i] >= camarilla_r3_aligned[i] and vol_confirmed
        elif is_trending:
            # In trending markets: breakout continuation at R4/S4
            long_signal = close[i] >= camarilla_r4_aligned[i] and vol_confirmed
            short_signal = close[i] <= camarilla_s4_aligned[i] and vol_confirmed
        
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

}