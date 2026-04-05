#!/usr/bin/env python3
"""
exp_7031_6h_donchian20_1d_pivot_vol_v2
Hypothesis: 6h Donchian(20) breakout with 1d pivot level confluence and volume confirmation.
Long when price breaks above Donchian(20) high AND above 1d R3 pivot level.
Short when price breaks below Donchian(20) low AND below 1d S3 pivot level.
Volume must be above 2.0x 20-period MA for confirmation.
Only trades in direction of 1d trend (price > 1d EMA50 for longs, price < 1d EMA50 for shorts).
Designed for 6h timeframe to capture institutional breakouts with ~12-37 trades/year.
Works in both bull and bear markets by aligning with 1d EMA50 trend filter.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_7031_6h_donchian20_1d_pivot_vol_v2"
timeframe = "6h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
VOL_MA_PERIOD = 20
VOL_BASE_THRESHOLD = 2.0
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5
MAX_HOLD_BARS = 30  # ~7.5 months (6h bars)
EMA_PERIOD = 50
PIVOT_LOOKBACK = 1  # Use previous day's pivot

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop - using 1d for pivot and EMA
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=EMA_PERIOD, adjust=False, min_periods=EMA_PERIOD).mean().values
    
    # Calculate 1d pivot points (using previous day's OHLC)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_prev = np.roll(close_1d, 1)  # Previous day close
    high_1d_prev = np.roll(high_1d, 1)    # Previous day high
    low_1d_prev = np.roll(low_1d, 1)      # Previous day low
    
    # Pivot point calculations
    pivot = (high_1d_prev + low_1d_prev + close_1d_prev) / 3.0
    r1 = 2 * pivot - low_1d_prev
    s1 = 2 * pivot - high_1d_prev
    r2 = pivot + (high_1d_prev - low_1d_prev)
    s2 = pivot - (high_1d_prev - low_1d_prev)
    r3 = high_1d_prev + 2 * (pivot - low_1d_prev)
    s3 = low_1d_prev - 2 * (high_1d_prev - pivot)
    r4 = r3 + (r2 - r1)
    s4 = s3 - (s2 - s1)
    
    # Align 1d indicators to 6h timeframe
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
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
    start = max(DONCHIAN_PERIOD, VOL_MA_PERIOD, ATR_PERIOD, EMA_PERIOD, PIVOT_LOOKBACK) + 1
    
    for i in range(start, n):
        bars_since_entry += 1
        
        # Skip if HTF data not available
        if np.isnan(ema_1d_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]):
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
        
        # Determine trend direction from 1d EMA50
        daily_uptrend = close[i] > ema_1d_aligned[i]
        daily_downtrend = close[i] < ema_1d_aligned[i]
        
        # Breakout signals with pivot confluence and trend filter
        long_breakout = daily_uptrend and (close[i] > highest_high[i]) and (close[i] > r3_aligned[i]) and vol_confirmed
        short_breakout = daily_downtrend and (close[i] < lowest_low[i]) and (close[i] < s3_aligned[i]) and vol_confirmed
        
        # Enter new positions only if flat
        if position == 0:
            if long_breakout:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                bars_since_entry = 0
            elif short_breakout:
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