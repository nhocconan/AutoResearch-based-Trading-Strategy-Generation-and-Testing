#!/usr/bin/env python3
"""
exp_7102_12h_camarilla1d_pivot_vol_v3
Hypothesis: 12h Camarilla pivot breakout with 1d volume confirmation and 1w trend filter.
Long when price breaks above H3/H4 in 1d uptrend (price > 1w EMA50), short when breaks below L3/L4 in 1d downtrend.
Uses volume spike to confirm breakout legitimacy. Designed for 12h to capture major swings with 12-37 trades/year.
Works in both bull and bear markets by aligning with 1w trend direction. Uses discrete position sizing (0.25) to minimize fee churn.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_7102_12h_camarilla1d_pivot_vol_v3"
timeframe = "12h"
leverage = 1.0

# Parameters
CAMARILLA_PERIOD = 1  # daily pivot
VOL_MA_PERIOD = 20
VOL_BASE_THRESHOLD = 2.0
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5
MAX_HOLD_BARS = 30  # ~30 * 12h = 15 days
EMA_PERIOD = 50

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop - using 1d for pivot, 1w for EMA
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=EMA_PERIOD, adjust=False, min_periods=EMA_PERIOD).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Calculate 1d Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    
    # Camarilla levels
    h5 = pivot + (range_1d * 1.5 / 2)
    h4 = pivot + (range_1d * 1.25 / 2)
    h3 = pivot + (range_1d * 1.1 / 2)
    l3 = pivot - (range_1d * 1.1 / 2)
    l4 = pivot - (range_1d * 1.25 / 2)
    l5 = pivot - (range_1d * 1.5 / 2)
    
    # Align to LTF (12h)
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4)
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
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
    start = max(CAMARILLA_PERIOD, VOL_MA_PERIOD, ATR_PERIOD, EMA_PERIOD) + 1
    
    for i in range(start, n):
        bars_since_entry += 1
        
        # Skip if HTF data not available
        if np.isnan(ema_1w_aligned[i]) or np.isnan(h4_aligned[i]):
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
        
        # Determine trend direction from 1w EMA50
        weekly_uptrend = close[i] > ema_1w_aligned[i]
        weekly_downtrend = close[i] < ema_1w_aligned[i]
        
        # Breakout signals aligned with 1w trend
        long_breakout = weekly_uptrend and (close[i] > h4_aligned[i]) and vol_confirmed
        short_breakout = weekly_downtrend and (close[i] < l4_aligned[i]) and vol_confirmed
        
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