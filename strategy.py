#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with volume confirmation and ATR-based stoploss.
Long when price breaks above 20-period high with volume > 1.5x average volume.
Short when price breaks below 20-period low with volume > 1.5x average volume.
Exit via ATR trailing stop (3x ATR) or opposite breakout.
Uses 1d EMA50 as trend filter: only take longs when price > 1d EMA50, shorts when price < 1d EMA50.
Target: 75-200 total trades over 4 years (19-50/year). Donchian breakouts capture momentum,
volume confirmation filters false breakouts, ATR stop manages risk, 1d EMA50 aligns with higher-timeframe trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 for trend filter
    close_1d_series = pd.Series(close_1d)
    ema50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate Donchian channels (20-period) on 4h
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Calculate average volume (20-period) for confirmation
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate ATR (14-period) for stoploss
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # Align 1d EMA50 to 4h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if np.isnan(ema50_1d_aligned[i]) or np.isnan(avg_volume[i]) or np.isnan(atr[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        ema50 = ema50_1d_aligned[i]
        atr_val = atr[i]
        
        if position == 0:
            # Long breakout: price > 20-period high AND volume > 1.5x avg volume AND price > 1d EMA50
            if price > highest_high[i] and vol > 1.5 * avg_volume[i] and price > ema50:
                signals[i] = 0.25
                position = 1
                entry_price = price
                highest_since_entry = price
            # Short breakout: price < 20-period low AND volume > 1.5x avg volume AND price < 1d EMA50
            elif price < lowest_low[i] and vol > 1.5 * avg_volume[i] and price < ema50:
                signals[i] = -0.25
                position = -1
                entry_price = price
                lowest_since_entry = price
        
        elif position == 1:
            # Update highest price since entry
            highest_since_entry = max(highest_since_entry, price)
            # ATR trailing stop: exit if price drops 3*ATR from highest since entry
            if price < highest_since_entry - 3.0 * atr_val:
                signals[i] = 0.0
                position = 0
            # Opposite breakout exit: price breaks below 20-period low
            elif price < lowest_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Update lowest price since entry
            lowest_since_entry = min(lowest_since_entry, price)
            # ATR trailing stop: exit if price rises 3*ATR from lowest since entry
            if price > lowest_since_entry + 3.0 * atr_val:
                signals[i] = 0.0
                position = 0
            # Opposite breakout exit: price breaks above 20-period high
            elif price > highest_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_VolumeConfirm_ATRStop_1dEMA50"
timeframe = "4h"
leverage = 1.0