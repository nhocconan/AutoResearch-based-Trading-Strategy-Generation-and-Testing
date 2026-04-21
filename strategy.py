#!/usr/bin/env python3
"""
1d_Donchian20_Breakout_VolumeChopRegime_ATRStop_v1
Hypothesis: Daily Donchian(20) breakouts with volume confirmation and choppy regime filter (CHOP>61.8) work in both bull and bear markets.
In ranging markets (CHOP>61.8), breakouts are faded; in trending markets (CHOP<38.2), breakouts are followed.
ATR-based trailing stop limits drawdown. Target: 15-25 trades/year per symbol (60-100 over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data once for chop regime filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate weekly chopiness index (EHLERS)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1w = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Sum of True Range over 14 periods
    sum_tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh_14 = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
    
    # Chopiness index: 100 * log10(sum_tr_14 / (hh_14 - ll_14)) / log10(14)
    range_14 = hh_14 - ll_14
    chop_raw = np.where(range_14 > 0, sum_tr_14 / range_14, np.nan)
    chop_1w = 100 * np.log10(chop_raw) / np.log10(14)
    
    # Align chop to daily timeframe (with 1-bar delay for completed weekly bar)
    chop_aligned = align_htf_to_ltf(prices, df_1w, chop_1w)
    
    # Load daily data for Donchian channels and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Donchian channels (20-period)
    highest_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian to daily timeframe (no extra delay needed)
    highest_20_aligned = align_htf_to_ltf(prices, df_1d, highest_20)
    lowest_20_aligned = align_htf_to_ltf(prices, df_1d, lowest_20)
    
    # Volume filter: 20-period average
    vol_ma = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # ATR for stoploss and position sizing (daily)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(chop_aligned[i]) or np.isnan(highest_20_aligned[i]) or 
            np.isnan(lowest_20_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        volume = prices['volume'].iloc[i]
        chop = chop_aligned[i]
        
        # Update trailing extremes
        if position == 1:
            highest_since_entry = max(highest_since_entry, price)
        elif position == -1:
            lowest_since_entry = min(lowest_since_entry, price)
        
        # Volume confirmation
        volume_ok = volume > 1.5 * vol_ma[i]
        
        if position == 0:
            # Regime-dependent entry logic
            if chop > 61.8:  # Ranging market: fade breakouts
                # Short at upper band, long at lower band
                if price > highest_20_aligned[i] and volume_ok:
                    signals[i] = -0.25
                    position = -1
                    entry_price = price
                    lowest_since_entry = price
                elif price < lowest_20_aligned[i] and volume_ok:
                    signals[i] = 0.25
                    position = 1
                    entry_price = price
                    highest_since_entry = price
            elif chop < 38.2:  # Trending market: follow breakouts
                # Long at upper band, short at lower band
                if price > highest_20_aligned[i] and volume_ok:
                    signals[i] = 0.25
                    position = 1
                    entry_price = price
                    highest_since_entry = price
                elif price < lowest_20_aligned[i] and volume_ok:
                    signals[i] = -0.25
                    position = -1
                    entry_price = price
                    lowest_since_entry = price
            # In transition zone (38.2 <= chop <= 61.8): no new entries
        
        elif position == 1:
            # Trailing stop: exit if price drops 2.5 * ATR from highest
            if price < highest_since_entry - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Trailing stop: exit if price rises 2.5 * ATR from lowest
            if price > lowest_since_entry + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_Breakout_VolumeChopRegime_ATRStop_v1"
timeframe = "1d"
leverage = 1.0