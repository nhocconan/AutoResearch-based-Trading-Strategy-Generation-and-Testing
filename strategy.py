#!/usr/bin/env python3
"""
exp_6751_6h_camarilla_pivot_1d_direction_v1
Hypothesis: 6h Camarilla pivot levels with 1d trend filter. At 1d uptrend: long at S3/S4 breakout, short at R3/R4 breakdown. At 1d downtrend: short at R3/R4 breakdown, long at S3/S4 breakout. Uses volume confirmation to avoid false breaks. Designed for low trade frequency (~15-35/year) with clear structural levels that work in both bull and bear markets by aligning with daily trend.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_6751_6h_camarilla_pivot_1d_direction_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
PIVOT_LOOKBACK = 1  # Use previous day's OHLC for Camarilla
VOL_MA_PERIOD = 20
VOL_CONFIRM_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0
MAX_HOLD_BARS = 8   # ~4 days (6h bars)
EMA_FAST = 9
EMA_SLOW = 21

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop - using 1d for trend and pivot calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMAs for trend filter
    close_1d = df_1d['close'].values
    ema_fast = pd.Series(close_1d).ewm(span=EMA_FAST, adjust=False, min_periods=EMA_FAST).mean().values
    ema_slow = pd.Series(close_1d).ewm(span=EMA_SLOW, adjust=False, min_periods=EMA_SLOW).mean().values
    
    # Align EMAs to LTF (6h)
    ema_fast_aligned = align_htf_to_ltf(prices, df_1d, ema_fast)
    ema_slow_aligned = align_htf_to_ltf(prices, df_1d, ema_slow)
    
    # Calculate previous day's Camarilla levels (using shift(1) for completed day only)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_prev = df_1d['close'].values
    
    # Pivot = (H + L + C) / 3
    pivot = (high_1d + low_1d + close_1d_prev) / 3.0
    range_hl = high_1d - low_1d
    
    # Camarilla levels
    r3 = pivot + (range_hl * 1.1 / 4)
    r4 = pivot + (range_hl * 1.1 / 2)
    s3 = pivot - (range_hl * 1.1 / 4)
    s4 = pivot - (range_hl * 1.1 / 2)
    
    # Align Camarilla levels to LTF
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
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
    
    # Volume MA for confirmation
    vol_ma = pd.Series(volume).rolling(window=VOL_MA_PERIOD, min_periods=VOL_MA_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = max(PIVOT_LOOKBACK, VOL_MA_PERIOD, ATR_PERIOD, EMA_SLOW) + 1
    
    for i in range(start, n):
        bars_since_entry += 1
        
        # Skip if HTF data not available
        if np.isnan(ema_fast_aligned[i]) or np.isnan(r3_aligned[i]):
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
        vol_confirmed = volume[i] > vol_ma[i] * VOL_CONFIRM_THRESHOLD if not np.isnan(vol_ma[i]) else False
        
        # Determine 1d trend direction
        daily_uptrend = ema_fast_aligned[i] > ema_slow_aligned[i]
        daily_downtrend = ema_fast_aligned[i] < ema_slow_aligned[i]
        
        # Entry logic based on daily trend and Camarilla levels
        long_signal = False
        short_signal = False
        
        if daily_uptrend:
            # In uptrend: look for longs at S3/S4 breakdown (mean reversion) or breakouts above R4
            if close[i] <= s3_aligned[i] and vol_confirmed:
                long_signal = True  # Mean reversion long at S3
            elif close[i] >= r4_aligned[i] and vol_confirmed:
                long_signal = True  # Breakout long above R4
        elif daily_downtrend:
            # In downtrend: look for shorts at R3/R4 bounce (mean reversion) or breakdowns below S4
            if close[i] >= r3_aligned[i] and vol_confirmed:
                short_signal = True  # Mean reversion short at R3
            elif close[i] <= s4_aligned[i] and vol_confirmed:
                short_signal = True  # Breakdown short below S4
        
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