#!/usr/bin/env python3
"""
exp_6539_6h_camarilla1d_pivot_vol_v1
Hypothesis: 6h Camarilla pivot levels from 1d timeframe for mean reversion at extreme levels (R3/S3, R4/S4) with volume confirmation.
In ranging markets: fade moves to R3/S3 (sell at R3, buy at S3) and R4/S4 (stronger fade).
In trending markets: breakout continuation when price closes beyond R4/S4 with volume spike.
Uses 1d Camarilla pivots calculated from prior 1d OHLC, aligned to 6h.
Designed for low-frequency, high-conviction trades targeting 75-200 total trades over 4 years.
Works in both bull and bear markets via mean reversion at extremes and breakout confirmation.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_6539_6h_camarilla1d_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
CAMARILLA_MULT = 1.1  # multiplier for R4/S4 levels
VOL_MA_PERIOD = 20
VOL_THRESHOLD = 2.0   # volume must be 2.0x its MA for confirmation
SIGNAL_SIZE = 0.25    # 25% position size

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop - using 1d for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d Camarilla pivot levels from prior day's OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Prior day's values (shift by 1 to avoid look-ahead)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Camarilla calculations
    range_1d = prev_high - prev_low
    camarilla_pivot = (prev_high + prev_low + prev_close) / 3
    camarilla_r3 = camarilla_pivot + range_1d * CAMARILLA_MULT * 1.1 / 4
    camarilla_s3 = camarilla_pivot - range_1d * CAMARILLA_MULT * 1.1 / 4
    camarilla_r4 = camarilla_pivot + range_1d * CAMARILLA_MULT * 1.5 / 4
    camarilla_s4 = camarilla_pivot - range_1d * CAMARILLA_MULT * 1.5 / 4
    
    # Align to LTF (6h) with shift(1) for completed bars only
    pivot_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot)
    r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume MA for confirmation
    vol_ma = pd.Series(volume).rolling(window=VOL_MA_PERIOD, min_periods=VOL_MA_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(VOL_MA_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(pivot_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or \
           np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]):
            continue
        
        # Mean reversion at R3/S3: sell at R3, buy at S3
        mr_long = close[i] < s3_aligned[i] and close[i-1] >= s3_aligned[i-1]  # cross below S3
        mr_short = close[i] > r3_aligned[i] and close[i-1] <= r3_aligned[i-1]  # cross above R3
        
        # Breakout continuation at R4/S4 with volume confirmation
        bo_long = close[i] > r4_aligned[i] and close[i-1] <= r4_aligned[i-1]  # cross above R4
        bo_short = close[i] < s4_aligned[i] and close[i-1] >= s4_aligned[i-1]  # cross below S4
        vol_confirm = volume[i] > vol_ma[i] * VOL_THRESHOLD if not np.isnan(vol_ma[i]) else False
        
        # Exit conditions: return to pivot or opposite extreme
        if position == 1:  # long position
            # Exit if price returns to pivot (mean reversion target)
            exit_long = close[i] > pivot_aligned[i]
            # Or if price reaches R4 (take profit)
            exit_long = exit_long or close[i] >= r4_aligned[i]
            if exit_long:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # short position
            # Exit if price returns to pivot (mean reversion target)
            exit_short = close[i] < pivot_aligned[i]
            # Or if price reaches S4 (take profit)
            exit_short = exit_short or close[i] <= s4_aligned[i]
            if exit_short:
                signals[i] = 0.0
                position = 0
                continue
        
        # Enter new positions only if flat
        if position == 0:
            if mr_long:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
            elif mr_short:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
            elif bo_long and vol_confirm:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
            elif bo_short and vol_confirm:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = position * SIGNAL_SIZE
    
    return signals