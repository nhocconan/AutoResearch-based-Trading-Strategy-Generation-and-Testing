#!/usr/bin/env python3
"""
4h Donchian(20) breakout with 1d EMA34 trend filter, volume spike confirmation, and ATR-based stop.
Hypothesis: Donchian channel breakouts capture strong momentum moves. Combined with daily EMA trend
filter and volume confirmation, this strategy works in both bull and bear markets by only taking
trend-aligned breakouts. Uses 4h timeframe with 1d HTF for trend and volume context.
Target: 75-200 trades over 4 years (19-50/year) to minimize fee drag.
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
    
    # Get 1d data for EMA trend and volume context (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:
        return np.zeros(n)
    
    # Calculate 34-period EMA on 1d close (only needs completed 1d candle)
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(
        span=34, adjust=False, min_periods=34
    ).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 20-period volume MA on 1d for volume spike reference
    vol_ma_20_1d = np.full(len(df_1d), np.nan)
    for i in range(20, len(df_1d)):
        vol_ma_20_1d[i] = np.mean(df_1d['volume'].values[i-19:i+1])
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Calculate ATR(14) on 4h for stoploss and position sizing reference
    tr = np.maximum(high[1:] - low[1:], np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])))
    tr = np.concatenate([[np.nan], tr])  # First bar has no TR
    atr = np.full(n, np.nan)
    for i in range(14, n):
        atr[i] = np.nanmean(tr[i-13:i+1])  # Simple average for ATR
    
    # Calculate Donchian channel (20-period high/low) on 4h
    donch_high = np.full(n, np.nan)
    donch_low = np.full(n, np.nan)
    for i in range(20, n):
        donch_high[i] = np.max(high[i-19:i+1])
        donch_low[i] = np.min(low[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for Donchian, EMA, volume MA, and ATR
    start_idx = max(20, 34, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(vol_ma_20_1d_aligned[i]) or 
            np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        ema_trend = ema_34_1d_aligned[i]
        upper_channel = donch_high[i]
        lower_channel = donch_low[i]
        vol_ma_1d = vol_ma_20_1d_aligned[i]
        atr_val = atr[i]
        
        # Volume confirmation: current 4h volume > 2.0 * 20-period 1d average volume (scaled)
        # Scale 1d volume to approximate 4h equivalent (1d = 6 * 4h bars)
        volume_confirm = curr_volume > 2.0 * (vol_ma_1d / 6.0)
        
        if position == 0:
            # Look for entry signals
            # Long: price breaks above Donchian upper channel, above 1d EMA, volume confirmation
            long_entry = (curr_close > upper_channel and 
                         curr_close > ema_trend and 
                         volume_confirm)
            # Short: price breaks below Donchian lower channel, below 1d EMA, volume confirmation
            short_entry = (curr_close < lower_channel and 
                          curr_close < ema_trend and 
                          volume_confirm)
            
            if long_entry:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_entry:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: price falls below Donchian lower channel OR below 1d EMA OR ATR stop (2.5 * ATR)
            atr_stop = entry_price - 2.5 * atr_val
            if curr_close < lower_channel or curr_close < ema_trend or curr_close < atr_stop:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price rises above Donchian upper channel OR above 1d EMA OR ATR stop (2.5 * ATR)
            atr_stop = entry_price + 2.5 * atr_val
            if curr_close > upper_channel or curr_close > ema_trend or curr_close > atr_stop:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Breakout_1dEMA34_Trend_VolumeSpike_ATRStop"
timeframe = "4h"
leverage = 1.0