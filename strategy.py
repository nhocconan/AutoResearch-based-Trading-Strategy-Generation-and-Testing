#!/usr/bin/env python3
"""
4h Donchian(20) breakout + volume confirmation + ATR stoploss + chop regime filter
Hypothesis: Donchian breakouts capture momentum bursts; volume confirms institutional participation;
choppiness regime filter avoids whipsaws in ranging markets. Works in bull/bear via ATR-based stops
and regime-adaptive sizing. Target: 20-50 trades/year on 4h timeframe.
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
    
    # Get 1d data for chop regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d True Range for chop regime
    if len(df_1d) >= 14:
        tr1 = pd.Series(df_1d['high']).diff().abs()
        tr2 = (pd.Series(df_1d['high']) - pd.Series(df_1d['close']).shift()).abs()
        tr3 = (pd.Series(df_1d['low']) - pd.Series(df_1d['close']).shift()).abs()
        tr_1d = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr_1d = tr_1d.rolling(window=14, min_periods=14).mean().values
    else:
        atr_1d = np.full(len(df_1d), 0.0)
    
    # Calculate 1d ADX for trend strength (optional, using chop as primary regime)
    # Chop = 100 * log10(sum(ATR14, n) / (max(high,n) - min(low,n))) / log10(n)
    # We'll use a simplified chop measure: ATR ratio to range
    if len(df_1d) >= 50:
        highest_50 = pd.Series(df_1d['high']).rolling(window=50, min_periods=50).max().values
        lowest_50 = pd.Series(df_1d['low']).rolling(window=50, min_periods=50).min().values
        atr_sum_50 = pd.Series(atr_1d).rolling(window=50, min_periods=50).sum().values
        range_50 = highest_50 - lowest_50
        # Avoid division by zero
        chop_raw = np.where(range_50 > 0, 100 * np.log10(atr_sum_50 / range_50) / np.log10(50), 50)
        chop_aligned = align_htf_to_ltf(prices, df_1d, chop_raw)
    else:
        chop_aligned = np.full(n, 50.0)
    
    # Calculate ATR(14) for 4h timeframe (for stoploss and Donchian width normalization)
    if len(close) >= 14:
        tr1 = pd.Series(high).diff().abs()
        tr2 = (pd.Series(high) - pd.Series(close).shift()).abs()
        tr3 = (pd.Series(low) - pd.Series(close).shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=14, min_periods=14).mean().values
    else:
        atr = np.full(n, 0.0)
    
    # Calculate Donchian channels (20-period) on 4h
    if len(close) >= 20:
        highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
        lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    else:
        highest_20 = np.full(n, np.nan)
        lowest_20 = np.full(n, np.nan)
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    else:
        vol_ma_20 = np.full(n, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    entry_bar = 0
    
    # Start index: need enough for all indicators
    start_idx = max(100, 50)  # ensure 1d chop has enough data
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(highest_20[i]) or 
            np.isnan(lowest_20[i]) or 
            np.isnan(vol_ma_20[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(chop_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        atr_val = atr[i]
        chop_val = chop_aligned[i]
        
        # Regime filter: chop < 61.8 = trending (favor breakouts), chop > 61.8 = ranging (avoid breakouts)
        # In ranging markets, we require stronger volume confirmation
        is_trending = chop_val < 61.8
        volume_spike = curr_volume > 1.5 * vol_ma_20[i]
        strong_volume = curr_volume > 2.0 * vol_ma_20[i]  # for ranging markets
        
        # Donchian breakout conditions
        long_breakout = curr_high > highest_20[i]
        short_breakout = curr_low < lowest_20[i]
        
        if position == 0:
            # Entry logic: breakout + volume confirmation + regime filter
            long_condition = long_breakout and volume_spike and (is_trending or strong_volume)
            short_condition = short_breakout and volume_spike and (is_trending or strong_volume)
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
                entry_bar = i
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
                entry_bar = i
        elif position == 1:
            # Exit conditions: stoploss, trend exhaustion, or time-based
            stoploss_level = entry_price - 2.5 * atr_val
            # Time exit: max 10 bars (~40 hours on 4h) to prevent stale positions
            time_exit = (i - entry_bar) >= 10
            # Trend exhaustion: price retracing back to middle of Donchian channel
            donchian_mid = (highest_20[i] + lowest_20[i]) / 2
            trend_exhaustion = curr_close < donchian_mid
            
            if curr_close <= stoploss_level or time_exit or trend_exhaustion:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit conditions for short
            stoploss_level = entry_price + 2.5 * atr_val
            time_exit = (i - entry_bar) >= 10
            donchian_mid = (highest_20[i] + lowest_20[i]) / 2
            trend_exhaustion = curr_close > donchian_mid
            
            if curr_close >= stoploss_level or time_exit or trend_exhaustion:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Breakout_Volume_ChopFilter_ATRStop_v1"
timeframe = "4h"
leverage = 1.0