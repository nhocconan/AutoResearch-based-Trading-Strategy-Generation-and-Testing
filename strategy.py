#!/usr/bin/env python3
"""
1d Donchian(20) Breakout + 1w EMA50 Trend + Volume Spike with ATR Trailing Stop
Hypothesis: Daily Donchian(20) breakouts capture significant price movements, 
filtered by weekly EMA50 trend to avoid counter-trend trades. Volume confirmation 
ensures breakout validity. ATR-based trailing stop manages risk. Designed to work 
in both bull markets (trend continuation) and bear markets (mean reversion to 
weekly EMA) by using discrete position sizing (0.25) to minimize fee drag and 
target 30-100 trades over 4 years (7-25/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_ema(series, period):
    """Calculate Exponential Moving Average"""
    if len(series) < period:
        return np.full_like(series, np.nan)
    return pd.Series(series).ewm(span=period, adjust=False, min_periods=period).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Weekly data for EMA50 trend (loaded ONCE)
    df_1w = get_htf_data(prices, '1w')
    
    # Weekly EMA50 trend filter
    ema_50_1w = calculate_ema(df_1w['close'].values, 50)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Daily Donchian(20) channels
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    # ATR for trailing stop (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_high_since_entry = 0.0
    lowest_low_since_entry = 0.0
    
    # Start index: need enough for Donchian, volume MA, and ATR
    start_idx = max(20, 20, 14) + 5
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        vol_spike = volume_spike[i]
        
        # Breakout conditions
        breakout_long = curr_close > donchian_high[i]
        breakout_short = curr_close < donchian_low[i]
        
        if position == 0:
            # Look for entry signals - require: Donchian breakout + volume spike + weekly EMA50 trend alignment
            long_entry = breakout_long and vol_spike and (curr_close > ema_50_1w_aligned[i])
            short_entry = breakout_short and vol_spike and (curr_close < ema_50_1w_aligned[i])
            
            if long_entry:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
                highest_high_since_entry = curr_high
                lowest_low_since_entry = curr_low
            elif short_entry:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
                highest_high_since_entry = curr_high
                lowest_low_since_entry = curr_low
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position: update highest high and check exit conditions
            highest_high_since_entry = max(highest_high_since_entry, curr_high)
            lowest_low_since_entry = min(lowest_low_since_entry, curr_low)
            
            # Exit conditions: retrace to Donchian low, trend change, or ATR trailing stop
            trailing_stop = highest_high_since_entry - 2.5 * atr[i]
            if curr_close < donchian_low[i] or curr_close < ema_50_1w_aligned[i] or curr_close < trailing_stop:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: update lowest low and check exit conditions
            highest_high_since_entry = max(highest_high_since_entry, curr_high)
            lowest_low_since_entry = min(lowest_low_since_entry, curr_low)
            
            # Exit conditions: retrace to Donchian high, trend change, or ATR trailing stop
            trailing_stop = lowest_low_since_entry + 2.5 * atr[i]
            if curr_close > donchian_high[i] or curr_close > ema_50_1w_aligned[i] or curr_close > trailing_stop:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_Breakout_1wEMA50_Trend_VolumeSpike_ATRTrailingStop"
timeframe = "1d"
leverage = 1.0