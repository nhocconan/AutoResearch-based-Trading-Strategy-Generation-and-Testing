#!/usr/bin/env python3
"""
exp_7255_6h_donchian20_1w_pivot_v1
Hypothesis: 6h Donchian(20) breakout with weekly pivot regime filter.
- In bull regime (price > weekly pivot): long Donchian breakouts, short breakdowns with confirmation
- In bear regime (price < weekly pivot): short Donchian breakdowns, long breakouts with confirmation
- Uses 1w pivot levels for trend regime and 6h volume for confirmation.
Designed for 6h timeframe to capture swings with ~12-37 trades/year (50-150 total over 4 years).
Weekly pivot provides structural support/resistance that works in both bull and bear markets.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_7255_6h_donchian20_1w_pivot_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
VOL_MA_PERIOD = 20
VOL_BASE_THRESHOLD = 1.8
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0
MAX_HOLD_BARS = 8  # ~4 days (8*6h = 48h)

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop - using 1w for pivot points
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot points (standard formula)
    # Pivot = (H + L + C) / 3
    # R1 = 2*P - L, S1 = 2*P - H
    # R2 = P + (H - L), S2 = P - (H - L)
    # R3 = H + 2*(P - L), S3 = L - 2*(H - P)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    r1_1w = 2 * pivot_1w - low_1w
    s1_1w = 2 * pivot_1w - high_1w
    r2_1w = pivot_1w + (high_1w - low_1w)
    s2_1w = pivot_1w - (high_1w - low_1w)
    r3_1w = high_1w + 2 * (pivot_1w - low_1w)
    s3_1w = low_1w - 2 * (high_1w - pivot_1w)
    
    # Align to LTF (6h)
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r3_1w_aligned = align_htf_to_ltf(prices, df_1w, r3_1w)
    s3_1w_aligned = align_htf_to_ltf(prices, df_1w, s3_1w)
    
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
        if np.isnan(pivot_1w_aligned[i]) or np.isnan(r3_1w_aligned[i]) or np.isnan(s3_1w_aligned[i]):
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
        above_pivot = close[i] > pivot_1w_aligned[i]
        below_pivot = close[i] < pivot_1w_aligned[i]
        
        # In bull regime (above weekly pivot): look for longs on breakouts, shorts on breakdowns with confirmation
        # In bear regime (below weekly pivot): look for shorts on breakdowns, longs on breakouts with confirmation
        if position == 0:
            # Bull regime logic
            if above_pivot:
                # Long breakout above Donchian high with volume
                if close[i] > highest_high[i] and vol_confirmed:
                    signals[i] = SIGNAL_SIZE
                    position = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                # Short breakdown below Donchian low with volume (counter-trend in bull)
                elif close[i] < lowest_low[i] and vol_confirmed:
                    signals[i] = -SIGNAL_SIZE
                    position = -1
                    entry_price = close[i]
                    bars_since_entry = 0
                else:
                    signals[i] = 0.0
            # Bear regime logic
            else:  # below_pivot
                # Short breakdown below Donchian low with volume
                if close[i] < lowest_low[i] and vol_confirmed:
                    signals[i] = -SIGNAL_SIZE
                    position = -1
                    entry_price = close[i]
                    bars_since_entry = 0
                # Long breakout above Donchian high with volume (counter-trend in bear)
                elif close[i] > highest_high[i] and vol_confirmed:
                    signals[i] = SIGNAL_SIZE
                    position = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                else:
                    signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = position * SIGNAL_SIZE
    
    return signals