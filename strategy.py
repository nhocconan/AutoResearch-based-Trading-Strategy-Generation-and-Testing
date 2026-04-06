#!/usr/bin/env python3
"""
12h Camarilla pivot with 1d volume confirmation and 1w trend filter.
- Long: price touches S3 + volume > 1.5x avg + price > 1w EMA(50)
- Short: price touches R3 + volume > 1.5x avg + price < 1w EMA(50)
- Exit: opposite pivot touch or stop loss (2*ATR)
- Position size: 0.25 (25%)
- Target: 50-150 trades over 4 years (12-37/year)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_14212_12h_camarilla_1d_vol_1w_ema_v1"
timeframe = "12h"
leverage = 1.0

def calculate_ema(close, period):
    """Calculate EMA with proper min_periods"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for volume average (once before loop)
    df_1d = get_htf_data(prices, '1d')
    vol_1d = df_1d['volume'].values
    
    # Load 1w data for EMA trend filter (once before loop)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate 1d 20-period volume average
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Calculate 1w EMA(50)
    ema_1w = calculate_ema(close_1w, 50)
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # 12h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla levels from previous 12h bar
    # HLC = (high + low + close) / 3
    hlc = (high + low + close) / 3.0
    # Shift to use previous bar's HLC (no look-ahead)
    hlc_prev = np.roll(hlc, 1)
    hlc_prev[0] = hlc[0]  # first bar uses its own
    
    # Calculate ranges
    rng = high - low
    
    # Camarilla levels
    s3 = hlc_prev - (rng * 1.1 / 4)
    s4 = hlc_prev - (rng * 1.1 / 2)
    r3 = hlc_prev + (rng * 1.1 / 4)
    r4 = hlc_prev + (rng * 1.1 / 2)
    
    # Volume filter: volume > 1.5x 1d average
    vol_filter = volume > (1.5 * vol_ma_1d_aligned)
    
    # ATR for stop loss (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period (max of 20 for vol, 14 for ATR, 1 for hlc)
    start = max(20, 14, 1) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(ema_1w_aligned[i]) or np.isnan(atr[i]) or np.isnan(vol_ma_1d_aligned[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check stops
        if position == 1:  # long position
            # Check stop loss
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # short position
            # Check stop loss
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Camarilla touch signals with volume and 1w EMA filter
        # Long: touch S3 + volume + price > 1w EMA
        # Short: touch R3 + volume + price < 1w EMA
        touch_long = (low[i] <= s3[i]) and vol_filter[i] and (close[i] > ema_1w_aligned[i])
        touch_short = (high[i] >= r3[i]) and vol_filter[i] and (close[i] < ema_1w_aligned[i])
        
        # Generate signals
        if position == 0:
            if touch_long:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (2.0 * atr[i])
            elif touch_short:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (2.0 * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long on stop or touch of R3
            if close[i] <= stop_price or high[i] >= r3[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short on stop or touch of S3
            if close[i] >= stop_price or low[i] <= s3[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals