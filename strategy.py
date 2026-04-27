#!/usr/bin/env python3
"""
1d_Donchian20_Breakout_WeeklyTrend_ATRStop_v1
Hypothesis: Daily Donchian(20) breakout with weekly trend filter (price above/below weekly EMA50) and ATR-based stoploss.
Volume confirmation ensures institutional participation. Designed for 10-25 trades/year to minimize fee drag.
Works in bull markets via breakouts and in bear markets via short breakdowns. Weekly trend filter avoids counter-trend trades.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate weekly EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Calculate daily Donchian channels (20-period)
    if len(prices) < 20:
        return np.zeros(n)
    
    # Rolling max/min for Donchian channels
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.5 * 20-day average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (1.5 * vol_avg)
    
    # ATR for stoploss (14-period on daily)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: need enough for all indicators
    start_idx = max(50, 20, 14)  # weekly EMA, Donchian, ATR
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_1w_aligned[i]) or np.isnan(highest_high[i]) or
            np.isnan(lowest_low[i]) or np.isnan(volume_confirmed[i]) or
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        atr_val = atr[i]
        weekly_trend = ema_1w_aligned[i]
        size = 0.25  # 25% position size to manage risk
        
        if position == 0:
            # Flat - look for breakout in direction of weekly trend with volume confirmation
            # Long: price above weekly EMA50 AND break above Donchian high + volume confirmation
            long_entry = (close_val > weekly_trend) and (close_val > highest_high[i]) and volume_confirmed[i]
            # Short: price below weekly EMA50 AND break below Donchian low + volume confirmation
            short_entry = (close_val < weekly_trend) and (close_val < lowest_low[i]) and volume_confirmed[i]
            
            if long_entry:
                signals[i] = size
                position = 1
                entry_price = close_val
            elif short_entry:
                signals[i] = -size
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit on Donchian low retracement or ATR stoploss
            exit_condition = (close_val < lowest_low[i]) or \
                           (close_val < entry_price - 2.5 * atr_val)
            if exit_condition:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit on Donchian high retracement or ATR stoploss
            exit_condition = (close_val > highest_high[i]) or \
                           (close_val > entry_price + 2.5 * atr_val)
            if exit_condition:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -size
    
    return signals

name = "1d_Donchian20_Breakout_WeeklyTrend_ATRStop_v1"
timeframe = "1d"
leverage = 1.0