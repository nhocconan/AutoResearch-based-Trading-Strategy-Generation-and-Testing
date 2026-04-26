#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_VolumeSpike_ChopRegime_v1
Hypothesis: 4h Donchian(20) breakout with volume confirmation and choppiness regime filter.
Long when price breaks above upper band AND volume > 1.5x MA AND chop > 61.8 (trending).
Short when price breaks below lower band AND volume > 1.5x MA AND chop > 61.8.
Uses ATR-based trailing stop for risk control. Designed for 75-200 trades over 4 years.
Works in bull markets (breakouts continue) and bear markets (breakdowns continue).
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
    
    # Get 1d data for ATR calculation (used for stoploss and choppiness)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate ATR(14) on 1d for stoploss and choppiness
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])  # align with 1d index
    
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    
    # Choppiness Index on 1d: CHOP = 100 * log10(sum(ATR,14) / (max(high,14) - min(low,14))) / log10(14)
    # We use a simplified version: CHOP > 61.8 = ranging, CHOP < 38.2 = trending
    # For trending regime filter, we want CHOP < 61.8 (not too choppy)
    atr_sum = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    max_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_sum / (max_high - min_low + 1e-10)) / np.log10(14)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Donchian channels on 4h (primary timeframe)
    # We calculate directly on 4h data since we're using 4h as primary TF
    # But we need to use rolling window on the 4h prices array
    # Since we can't call get_htf_data for 4h inside loop (it's the primary), we calculate directly
    lookback = 20
    # Upper band: highest high over last 20 periods
    upper_band = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    # Lower band: lowest low over last 20 periods
    lower_band = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Warmup: max of Donchian lookback, volume MA, ATR calculation
    start_idx = max(lookback, 20, 14) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or
            np.isnan(vol_ma[i]) or np.isnan(atr_14_aligned[i]) or
            np.isnan(chop_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        vol_conf = volume_confirm[i]
        # Trending regime: chop < 61.8 (not too choppy)
        trending_regime = chop_aligned[i] < 61.8
        atr_val = atr_14_aligned[i]
        
        if position == 0:
            # Long: price breaks above upper band AND volume confirm AND trending regime
            long_signal = (close_val > upper_band[i]) and vol_conf and trending_regime
            
            # Short: price breaks below lower band AND volume confirm AND trending regime
            short_signal = (close_val < lower_band[i]) and vol_conf and trending_regime
            
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
            highest_since_entry = max(highest_since_entry, high_val)
            
            # Exit conditions:
            # 1. ATR trailing stop: price drops below highest - 2.5 * ATR
            # 2. Price re-enters Donchian channel (failed breakout)
            # 3. Regime becomes too choppy
            trailing_stop = highest_since_entry - (2.5 * atr_val)
            re_entry = close_val < upper_band[i]  # price back below upper band
            choppy_regime = chop_aligned[i] >= 61.8
            
            if (close_val < trailing_stop) or re_entry or choppy_regime:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            lowest_since_entry = min(lowest_since_entry, low_val)
            
            # Exit conditions:
            # 1. ATR trailing stop: price rises above lowest + 2.5 * ATR
            # 2. Price re-enters Donchian channel (failed breakdown)
            # 3. Regime becomes too choppy
            trailing_stop = lowest_since_entry + (2.5 * atr_val)
            re_entry = close_val > lower_band[i]  # price back above lower band
            choppy_regime = chop_aligned[i] >= 61.8
            
            if (close_val > trailing_stop) or re_entry or choppy_regime:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Donchian20_Breakout_VolumeSpike_ChopRegime_v1"
timeframe = "4h"
leverage = 1.0