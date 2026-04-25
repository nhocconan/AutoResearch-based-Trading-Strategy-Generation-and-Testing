#!/usr/bin/env python3
"""
4h_Donchian20_Volume_Trend_ATRStop
Hypothesis: On 4h timeframe, enter long when price breaks above Donchian(20) high with volume confirmation and 1d EMA34 uptrend; enter short when price breaks below Donchian(20) low with volume confirmation and 1d EMA34 downtrend. Exit via ATR-based trailing stop (3*ATR) or opposite signal. Designed for low trade frequency (target: 75-200 total trades over 4 years) to minimize fee drag. Works in both bull and bear markets by using 1d trend filter and volatility-adjusted stops.
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
    
    # 1d data for EMA trend filter (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA34 for trend filter
    close_1d = pd.Series(df_1d['close'].values)
    ema_34_1d = close_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Donchian(20) channels on 4h
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Start index: need enough for 1d EMA (34) and Donchian (20)
    start_idx = 34
    
    for i in range(start_idx, n):
        # Update trailing stop levels
        if position == 1:
            highest_since_entry = max(highest_since_entry, high[i])
        elif position == -1:
            lowest_since_entry = min(lowest_since_entry, low[i])
        
        # Skip if any data not ready
        if (np.isnan(ema_34_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(atr[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Check for stoploss hit
        if position == 1:
            if curr_close <= highest_since_entry - 3.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:
            if curr_close >= lowest_since_entry + 3.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                continue
        
        if position == 0:
            # Look for entry signals
            # Long: price breaks above Donchian high + volume + 1d uptrend
            long_breakout = curr_close > highest_high[i]
            long_trend = curr_close > ema_34_aligned[i]
            long_volume = volume_spike[i]
            
            # Short: price breaks below Donchian low + volume + 1d downtrend
            short_breakout = curr_close < lowest_low[i]
            short_trend = curr_close < ema_34_aligned[i]
            short_volume = volume_spike[i]
            
            if long_breakout and long_trend and long_volume:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
                highest_since_entry = curr_high
                lowest_since_entry = curr_low
            elif short_breakout and short_trend and short_volume:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
                highest_since_entry = curr_high
                lowest_since_entry = curr_low
            else:
                signals[i] = 0.0
        elif position == 1:
            # Stay long until stop or reversal signal
            signals[i] = 0.25
        elif position == -1:
            # Stay short until stop or reversal signal
            signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Volume_Trend_ATRStop"
timeframe = "4h"
leverage = 1.0