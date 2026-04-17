#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_With_Volume_and_Regime_v1
Breakout from Donchian(20) channels with volume confirmation and chop regime filter.
Long: price breaks above upper band, volume > 1.5x avg, CHOP > 61.8 (range).
Short: price breaks below lower band, volume > 1.5x avg, CHOP > 61.8.
Exit: price returns to middle band or CHOP < 38.2 (trend).
Uses 1d ADX(14) > 20 as additional trend filter (avoid chop).
Designed to capture breakouts in ranging markets with volume confirmation.
Target: 20-60 total trades over 4 years (5-15/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === Donchian Channels (20) ===
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    middle_band = (highest_high + lowest_low) / 2.0
    
    # === Volume Filter ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === Choppiness Index (14) ===
    atr1 = np.abs(high - low)
    atr2 = np.abs(high - np.roll(close, 1))
    atr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(atr1, np.maximum(atr2, atr3))
    tr[0] = atr1[0]
    
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    hh = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_sum / (hh - ll)) / np.log10(14)
    chop = np.where((hh - ll) == 0, 50, chop)  # avoid div by zero
    
    # === 1d ADX(14) for trend filter ===
    df_1d = get_htf_data(prices, '1d')
    d1_high = df_1d['high'].values
    d1_low = df_1d['low'].values
    d1_close = df_1d['close'].values
    
    plus_dm = np.where((d1_high[1:] - d1_high[:-1]) > (d1_low[:-1] - d1_low[1:]), 
                       np.maximum(d1_high[1:] - d1_high[:-1], 0), 0)
    minus_dm = np.where((d1_low[:-1] - d1_low[1:]) > (d1_high[1:] - d1_high[:-1]), 
                        np.maximum(d1_low[:-1] - d1_low[1:], 0), 0)
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    
    tr1 = d1_high - d1_low
    tr2 = np.abs(d1_high - np.roll(d1_close, 1))
    tr3 = np.abs(d1_low - np.roll(d1_close, 1))
    tr_d1 = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_d1[0] = tr1[0]
    
    atr_d1 = pd.Series(tr_d1).rolling(window=14, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).sum().values / (atr_d1 * 14)
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).sum().values / (atr_d1 * 14)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx_1d = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 60
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or 
            np.isnan(vol_ma[i]) or 
            np.isnan(chop[i]) or 
            np.isnan(adx_1d_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: breakout above upper band, volume spike, chop > 61.8 (range), ADX > 20
            if (close[i] > highest_high[i] and 
                volume[i] > 1.5 * vol_ma[i] and 
                chop[i] > 61.8 and 
                adx_1d_aligned[i] > 20):
                signals[i] = 0.25
                position = 1
                continue
            # Short: breakdown below lower band, volume spike, chop > 61.8 (range), ADX > 20
            elif (close[i] < lowest_low[i] and 
                  volume[i] > 1.5 * vol_ma[i] and 
                  chop[i] > 61.8 and 
                  adx_1d_aligned[i] > 20):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: return to middle band OR chop < 38.2 (trend)
            if (close[i] <= middle_band[i] or 
                chop[i] < 38.2):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: return to middle band OR chop < 38.2 (trend)
            if (close[i] >= middle_band[i] or 
                chop[i] < 38.2):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Breakout_With_Volume_and_Regime_v1"
timeframe = "4h"
leverage = 1.0