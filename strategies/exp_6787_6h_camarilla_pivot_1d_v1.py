#!/usr/bin/env python3
"""
exp_6787_6h_camarilla_pivot_1d_v1
Hypothesis: 6h Camarilla pivot levels from 1d timeframe. Fade at R3/S3 (mean reversion in range), 
breakout continuation at R4/S4 (trend following). Uses volume confirmation to filter false signals.
Designed for 6h timeframe to capture intermediate swings with ~12-37 trades/year (50-150 total over 4 years).
Works in both bull and bear markets by adapting to price action relative to daily pivot structure.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_6787_6h_camarilla_pivot_1d_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
PIVOT_LOOKBACK = 1  # Use previous day's OHLC for Camarilla calculation
VOL_MA_PERIOD = 20
VOL_BASE_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0
MAX_HOLD_BARS = 8  # ~4 days (6h bars)
CHOP_PERIOD = 14
CHOP_THRESHOLD = 61.8  # Above = range (mean revert), Below = trend (breakout)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop - using 1d for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla pivot levels from previous day's OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: based on previous day's range
    R4 = close_1d + (high_1d - low_1d) * 1.1 / 2
    R3 = close_1d + (high_1d - low_1d) * 1.1 / 4
    R2 = close_1d + (high_1d - low_1d) * 1.1 / 6
    R1 = close_1d + (high_1d - low_1d) * 1.1 / 12
    S1 = close_1d - (high_1d - low_1d) * 1.1 / 12
    S2 = close_1d - (high_1d - low_1d) * 1.1 / 6
    S3 = close_1d - (high_1d - low_1d) * 1.1 / 4
    S4 = close_1d - (high_1d - low_1d) * 1.1 / 2
    Pivot = (high_1d + low_1d + close_1d) / 3
    
    # Align to LTF (6h)
    R4_aligned = align_htf_to_ltf(prices, df_1d, R4)
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    R2_aligned = align_htf_to_ltf(prices, df_1d, R2)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    S2_aligned = align_htf_to_ltf(prices, df_1d, S2)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    S4_aligned = align_htf_to_ltf(prices, df_1d, S4)
    Pivot_aligned = align_htf_to_ltf(prices, df_1d, Pivot)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Choppiness Index to determine market regime
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_sum = tr.rolling(window=CHOP_PERIOD, min_periods=CHOP_PERIOD).sum()
    highest_high = pd.Series(high).rolling(window=CHOP_PERIOD, min_periods=CHOP_PERIOD).max()
    lowest_low = pd.Series(low).rolling(window=CHOP_PERIOD, min_periods=CHOP_PERIOD).min()
    chop = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(CHOP_PERIOD)
    chop_values = chop.values
    
    # ATR for stoploss
    tr_atr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr_atr.ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    # Volume MA for confirmation
    vol_ma = pd.Series(volume).rolling(window=VOL_MA_PERIOD, min_periods=VOL_MA_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = max(PIVOT_LOOKBACK, VOL_MA_PERIOD, ATR_PERIOD, CHOP_PERIOD) + 1
    
    for i in range(start, n):
        bars_since_entry += 1
        
        # Skip if HTF data not available
        if np.isnan(Pivot_aligned[i]):
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
        
        # Determine market regime from Choppiness Index
        is_range = chop_values[i] > CHOP_THRESHOLD if not np.isnan(chop_values[i]) else True
        is_trend = chop_values[i] <= CHOP_THRESHOLD if not np.isnan(chop_values[i]) else False
        
        # Camarilla-based signals
        if is_range:
            # Range market: mean reversion at R3/S3
            long_signal = close[i] <= S3_aligned[i] and vol_confirmed
            short_signal = close[i] >= R3_aligned[i] and vol_confirmed
        else:
            # Trending market: breakout continuation at R4/S4
            long_signal = close[i] >= R4_aligned[i] and vol_confirmed
            short_signal = close[i] <= S4_aligned[i] and vol_confirmed
        
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