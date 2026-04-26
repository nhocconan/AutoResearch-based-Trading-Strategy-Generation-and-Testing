#!/usr/bin/env python3
"""
6h_WeeklyDonchian_Breakout_1dTrend_VolumeConfirm_v1
Hypothesis: On 6h timeframe, trade breakouts above/below weekly Donchian(20) channels only when aligned with 1d EMA50 trend and confirmed by volume spike (>2.0x 24-bar average). Uses ATR(14) stoploss at 2.0x ATR. Discrete sizing at 0.25 to limit fee drag. Target: 12-30 trades/year on BTC/ETH/SOL.
Weekly Donchian provides structural weekly support/resistance, 1d EMA50 filters for intermediate trend, volume confirms institutional participation. Works in both bull (breakouts with trend) and bear (failed breaks reverse) markets.
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
    
    # Get weekly data for Donchian channels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate weekly Donchian(20) - upper/lower bounds
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Rolling window of 20 weeks for Donchian channels
    upper_20 = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    lower_20 = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Align all HTF indicators to 6h timeframe
    upper_20_aligned = align_htf_to_ltf(prices, df_1w, upper_20)
    lower_20_aligned = align_htf_to_ltf(prices, df_1w, lower_20)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # ATR for stoploss calculation (6h ATR)
    atr_period = 14
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]  # first bar
    atr = pd.Series(tr).ewm(span=atr_period, min_periods=atr_period, adjust=False).mean().values
    
    # Volume spike: current volume > 2.0 * 24-period average (4 days on 6h)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: max of Donchian calc (20), EMA50 (50), ATR (14), volume MA (24)
    start_idx = max(20, 50, 14, 24) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(upper_20_aligned[i]) or 
            np.isnan(lower_20_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(atr[i]) or
            np.isnan(vol_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        upper_val = upper_20_aligned[i]
        lower_val = lower_20_aligned[i]
        ema_50_val = ema_50_1d_aligned[i]
        close_val = close[i]
        atr_val = atr[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long: price breaks above weekly upper Donchian, above 1d EMA50, with volume spike
            long_signal = (close_val > upper_val) and (close_val > ema_50_val) and vol_spike
            
            # Short: price breaks below weekly lower Donchian, below 1d EMA50, with volume spike
            short_signal = (close_val < lower_val) and (close_val < ema_50_val) and vol_spike
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price breaks below weekly lower Donchian OR ATR stoploss (2.0*ATR below entry)
            if (close_val < lower_val) or (close_val < entry_price - 2.0 * atr_val):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price breaks above weekly upper Donchian OR ATR stoploss (2.0*ATR above entry)
            if (close_val > upper_val) or (close_val > entry_price + 2.0 * atr_val):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_WeeklyDonchian_Breakout_1dTrend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0