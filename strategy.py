#!/usr/bin/env python3
"""
12h Donchian20 Breakout + 1w EMA50 Trend + Volume Spike
Hypothesis: Donchian(20) breakouts on 12h aligned with 1w EMA50 trend and volume confirmation
capture strong momentum moves while filtering noise. Works in bull/bear by following weekly trend.
Uses discrete position sizing (0.25) and volume threshold (2.0x) to target 50-150 total trades over 4 years (12-37/year).
ATR-based trailing stop reduces drawdown. 12h timeframe avoids overtrading fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Donchian(20) from 12h data
    lookback = 20
    highest_20 = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_20 = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # 1w EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # 1d ATR for dynamic stoploss
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # 12h volume average for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Start index: need enough for Donchian20 + EMA50(1w) + ATR(1d) + VolMA20
    start_idx = max(lookback, 50, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(atr_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema_50_level = ema_50_1w_aligned[i]
        atr_value = atr_1d_aligned[i]
        
        # Volume spike: current volume > 2.0 * 20-period average
        volume_spike = curr_volume > 2.0 * vol_ma_20[i]
        
        # Breakout conditions
        bullish_breakout = curr_close > highest_20[i]
        bearish_breakout = curr_close < lowest_20[i]
        
        # Update tracking variables for trailing stop
        if position == 1:
            highest_since_entry = max(highest_since_entry, curr_high)
        elif position == -1:
            lowest_since_entry = min(lowest_since_entry, curr_low)
        
        # Exit conditions: trailing stop or reverse breakout or trend change
        if position != 0:
            exit_signal = False
            
            if position == 1:
                # Trailing stop: exit if price drops 2.5*ATR from highest since entry
                if curr_close < highest_since_entry - 2.5 * atr_value:
                    exit_signal = True
                # Reverse breakout or trend change
                elif curr_close < lowest_20[i] or curr_close < ema_50_level:
                    exit_signal = True
                    
            elif position == -1:
                # Trailing stop: exit if price rises 2.5*ATR from lowest since entry
                if curr_close > lowest_since_entry + 2.5 * atr_value:
                    exit_signal = True
                # Reverse breakout or trend change
                elif curr_close > highest_20[i] or curr_close > ema_50_level:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
                continue
        
        # Entry conditions: Breakout + trend + volume
        if position == 0:
            long_condition = bullish_breakout and (curr_close > ema_50_level) and volume_spike
            short_condition = bearish_breakout and (curr_close < ema_50_level) and volume_spike
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                highest_since_entry = curr_high
                lowest_since_entry = curr_low
            elif short_condition:
                signals[i] = -0.25
                position = -1
                highest_since_entry = curr_high
                lowest_since_entry = curr_low
        elif position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_Breakout_1wEMA50_Trend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0