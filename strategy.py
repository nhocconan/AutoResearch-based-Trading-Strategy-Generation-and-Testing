#!/usr/bin/env python3
"""
1d_Camarilla_R1S1_Breakout_1wTrend_v3
Hypothesis: Trade daily Camarilla R1/S1 breakouts with weekly EMA50 trend filter and ATR-based stoploss.
Uses volume confirmation and choppiness regime filter to avoid whipsaws. Targets 30-100 trades over 4 years
(7-25/year) for 1d timeframe to minimize fee drag. Weekly trend filter provides strong directional bias
that works in both bull and bear markets by aligning with higher timeframe momentum.
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
    
    # Get 1w data for weekly trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate EMA(50) on 1w for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate ATR(14) for stoploss
    tr1 = np.maximum(high[1:], close[:-1]) - np.minimum(low[1:], close[:-1])
    tr1 = np.concatenate([[0], tr1])  # align length
    atr = pd.Series(tr1).rolling(window=14, min_periods=14).mean().values
    
    # Get 1d data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 1d bar
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Avoid NaN from shift
    prev_high = np.where(np.isnan(prev_high), df_1d['high'].values, prev_high)
    prev_low = np.where(np.isnan(prev_low), df_1d['low'].values, prev_low)
    prev_close = np.where(np.isnan(prev_close), df_1d['close'].values, prev_close)
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_hl = prev_high - prev_low
    r1 = pivot + (range_hl * 1.1 / 12.0)
    s1 = pivot - (range_hl * 1.1 / 12.0)
    
    # Align Camarilla levels to 1d
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    # Choppiness regime filter: avoid ranging markets
    lookback = 14
    tr_sum = pd.Series(tr1).rolling(window=lookback, min_periods=lookback).sum().values
    max_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    min_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    chop_denom = max_high - min_low
    chop_denom = np.where(chop_denom == 0, 1, chop_denom)  # avoid div by zero
    chopiness = 100 * np.log10(tr_sum / chop_denom) / np.log10(lookback)
    # Market is trending when CHOP < 38.2, ranging when CHOP > 61.8
    chop_filter = chopiness < 50.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Warmup: max of 1w EMA(50), volume MA(20), ATR(14), chop lookback(14), and need 1d data
    start_idx = max(50, 20, 14) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(atr[i]) or
            np.isnan(vol_ma[i]) or
            np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i]) or
            np.isnan(chop_filter[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        close_val = close[i]
        vol_conf = volume_confirm[i]
        chop_ok = chop_filter[i]
        trend_up = close_val > ema_50_1w_aligned[i]   # 1w uptrend
        trend_down = close_val < ema_50_1w_aligned[i]  # 1w downtrend
        
        if position == 0:
            # Long: price breaks above R1 AND volume confirm AND chop OK AND 1w uptrend
            long_signal = (close_val > r1_aligned[i]) and vol_conf and chop_ok and trend_up
            
            # Short: price breaks below S1 AND volume confirm AND chop OK AND 1w downtrend
            short_signal = (close_val < s1_aligned[i]) and vol_conf and chop_ok and trend_down
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
                highest_since_entry = close_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
                lowest_since_entry = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            highest_since_entry = max(highest_since_entry, close_val)
            # ATR trailing stop: exit if price drops 2.0 * ATR from highest since entry
            if close_val < highest_since_entry - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Alternative exit: 1w trend flips down OR chop too high
            elif (not trend_up) or (not chop_ok):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            lowest_since_entry = min(lowest_since_entry, close_val)
            # ATR trailing stop: exit if price rises 2.0 * ATR from lowest since entry
            if close_val > lowest_since_entry + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Alternative exit: 1w trend flips up OR chop too high
            elif (not trend_down) or (not chop_ok):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Camarilla_R1S1_Breakout_1wTrend_v3"
timeframe = "1d"
leverage = 1.0