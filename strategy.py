#!/usr/bin/env python3
"""
1d_Donchian20_Breakout_1wTrend_VolumeSpike_v1
Hypothesis: Daily Donchian(20) breakouts with 1-week trend filter (price > EMA50 = uptrend) and volume spike confirmation. Uses ATR trailing stop and trend reversal exit. Designed for low trade frequency (~15-25/year) to minimize fee drag. Volume threshold at 2.0x average balances signal quality and trade frequency. Works in both bull and bear markets by requiring trend alignment and volatility expansion.
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
    
    # Get 1d data for Donchian calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 20-period Donchian channels on 1d
    highest_20 = pd.Series(df_1d['high'].values).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(df_1d['low'].values).rolling(window=20, min_periods=20).min().values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1w EMA(50) for trend filter
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # ATR for stop (14-period on 1d)
    tr1_1d = df_1d['high'].values[1:] - df_1d['low'].values[1:]
    tr2_1d = np.abs(df_1d['high'].values[1:] - df_1d['close'].values[:-1])
    tr3_1d = np.abs(df_1d['low'].values[1:] - df_1d['close'].values[:-1])
    tr_1d = np.concatenate([[np.nan], np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))])
    atr_14_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    
    # Volume confirmation: 2.0x average volume (balanced for trade frequency)
    vol_ma_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    
    # Align 1d indicators to 1d timeframe (identity alignment)
    highest_20_aligned = align_htf_to_ltf(prices, df_1d, highest_20)
    lowest_20_aligned = align_htf_to_ltf(prices, df_1d, lowest_20)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Align 1w indicators to 1d timeframe
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    long_stop = 0.0
    short_stop = 0.0
    
    # Warmup: max of Donchian (20), volume MA (20), ATR (14), 1w EMA (50)
    start_idx = max(20, 20, 14, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(highest_20_aligned[i]) or 
            np.isnan(lowest_20_aligned[i]) or 
            np.isnan(vol_ma_1d_aligned[i]) or 
            np.isnan(atr_14_1d_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        highest_20_val = highest_20_aligned[i]
        lowest_20_val = lowest_20_aligned[i]
        vol_ma_val = vol_ma_1d_aligned[i]
        atr_val = atr_14_1d_aligned[i]
        ema_50_1w_val = ema_50_1w_aligned[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        volume_val = volume[i]
        
        if position == 0:
            # Long: break above upper Donchian, uptrend (close > EMA50), volume spike
            long_signal = (high_val > highest_20_val) and (close_val > ema_50_1w_val) and (volume_val > 2.0 * vol_ma_val)
            # Short: break below lower Donchian, downtrend (close < EMA50), volume spike
            short_signal = (low_val < lowest_20_val) and (close_val < ema_50_1w_val) and (volume_val > 2.0 * vol_ma_val)
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
                long_stop = entry_price - 1.5 * atr_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
                short_stop = entry_price + 1.5 * atr_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Update trailing stop: move stop up as price makes new highs
            long_stop = max(long_stop, high_val - 1.5 * atr_val)
            # Exit: trailing stop hit or trend reversal (close < EMA50)
            if (low_val < long_stop) or (close_val < ema_50_1w_val):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Update trailing stop: move stop down as price makes new lows
            short_stop = min(short_stop, low_val + 1.5 * atr_val)
            # Exit: trailing stop hit or trend reversal (close > EMA50)
            if (high_val > short_stop) or (close_val > ema_50_1w_val):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Donchian20_Breakout_1wTrend_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0