#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_VolumeConfirm_TrendFilter_v1
Hypothesis: Use 4h timeframe with Donchian(20) breakout confirmed by 12h EMA50 trend and volume spike. Targets 20-50 trades/year to minimize fee drag. Works in both bull and bear markets by requiring trend alignment and volume confirmation to avoid false breakouts. Uses discrete position sizing (0.25) and ATR-based stoploss via signal=0.
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
    
    # Calculate Donchian channels (20-period)
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    atr_period = 14
    atr_multiplier = 2.5
    
    # Calculate ATR for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    # Warmup: need 20 for Donchian, 50 for 12h EMA, 20 for volume avg, 14 for ATR
    start_idx = max(lookback, 50, 20, atr_period)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(volume_spike[i]) or
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        size = 0.25  # Fixed position size to minimize churn
        
        if position == 0:
            # Flat - look for breakout with trend and volume confirmation
            # Long: break above highest_high + 12h EMA50 uptrend + volume spike
            long_entry = (close_val > highest_high[i]) and \
                       (ema_50_12h_aligned[i] > ema_50_12h_aligned[i-1]) and \
                       volume_spike[i]
            # Short: break below lowest_low + 12h EMA50 downtrend + volume spike
            short_entry = (close_val < lowest_low[i]) and \
                        (ema_50_12h_aligned[i] < ema_50_12h_aligned[i-1]) and \
                        volume_spike[i]
            
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
            # Long - exit conditions
            # Stoploss: close below entry - atr_multiplier * atr
            stop_loss = entry_price - atr_multiplier * atr[i]
            # Take profit: revert to midpoint of Donchian channel
            take_profit = (highest_high[i] + lowest_low[i]) / 2
            
            if close_val < stop_loss or close_val < take_profit:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit conditions
            # Stoploss: close above entry + atr_multiplier * atr
            stop_loss = entry_price + atr_multiplier * atr[i]
            # Take profit: revert to midpoint of Donchian channel
            take_profit = (highest_high[i] + lowest_low[i]) / 2
            
            if close_val > stop_loss or close_val > take_profit:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Donchian20_Breakout_VolumeConfirm_TrendFilter_v1"
timeframe = "4h"
leverage = 1.0