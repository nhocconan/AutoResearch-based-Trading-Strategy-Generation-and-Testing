#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_Trend_Volume
Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation.
Long when: price breaks above 20-period high + 1d EMA50 uptrend + volume > 1.3 * avg volume.
Short when: price breaks below 20-period low + 1d EMA50 downtrend + volume > 1.3 * avg volume.
Exit: ATR-based trailing stop (3 * ATR) or time-based exit after 20 bars.
Designed for BTC/ETH: avoids overtrading with tight entry conditions, works in trending and ranging markets via volume/spike filter.
Targets ~25-40 trades/year per symbol for optimal test generalization.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: current volume > 1.3 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.3 * vol_avg)
    
    # ATR for dynamic stoploss (14-period)
    atr_period = 14
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    bars_since_entry = 0
    
    # Warmup: need 20 for Donchian, 50 for EMA, 20 for volume avg, 14 for ATR
    start_idx = max(lookback, 50, 20, atr_period)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_confirm[i]) or
            np.isnan(atr[i])):
            signals[i] = 0.0
            if position != 0:
                bars_since_entry += 1
            continue
        
        close_val = close[i]
        size = 0.25  # Fixed position size (25% of capital)
        
        if position == 0:
            # Flat - look for breakout with trend and volume confirmation
            # Long: break above 20-period high + 1d EMA50 uptrend + volume confirmation
            long_entry = (close_val > highest_high[i]) and \
                       (ema_50_1d_aligned[i] > ema_50_1d_aligned[i-1]) and \
                       volume_confirm[i]
            # Short: break below 20-period low + 1d EMA50 downtrend + volume confirmation
            short_entry = (close_val < lowest_low[i]) and \
                        (ema_50_1d_aligned[i] < ema_50_1d_aligned[i-1]) and \
                        volume_confirm[i]
            
            if long_entry:
                signals[i] = size
                position = 1
                entry_price = close_val
                highest_since_entry = close_val
                lowest_since_entry = close_val
                bars_since_entry = 0
            elif short_entry:
                signals[i] = -size
                position = -1
                entry_price = close_val
                highest_since_entry = close_val
                lowest_since_entry = close_val
                bars_since_entry = 0
            else:
                signals[i] = 0.0
                bars_since_entry = 0
        elif position == 1:
            # Long - update highest since entry
            highest_since_entry = max(highest_since_entry, close_val)
            lowest_since_entry = min(lowest_since_entry, close_val)
            
            # Exit conditions:
            # 1. ATR trailing stop: price drops 3*ATR from highest since entry
            atr_stop = highest_since_entry - 3.0 * atr[i]
            # 2. Time-based exit: exit after 20 bars to avoid overtrading
            time_exit = bars_since_entry >= 20
            
            if close_val < atr_stop or time_exit:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = size
                bars_since_entry += 1
        elif position == -1:
            # Short - update lowest since entry
            lowest_since_entry = min(lowest_since_entry, close_val)
            highest_since_entry = max(highest_since_entry, close_val)
            
            # Exit conditions:
            # 1. ATR trailing stop: price rises 3*ATR from lowest since entry
            atr_stop = lowest_since_entry + 3.0 * atr[i]
            # 2. Time-based exit: exit after 20 bars to avoid overtrading
            time_exit = bars_since_entry >= 20
            
            if close_val > atr_stop or time_exit:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -size
                bars_since_entry += 1
    
    return signals

name = "4h_Donchian20_Breakout_Trend_Volume"
timeframe = "4h"
leverage = 1.0