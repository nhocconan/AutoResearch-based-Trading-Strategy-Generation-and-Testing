#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Donchian20_Volume_Spike_Regime_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily trend filter: EMA34
    df_1d = get_htf_data(prices, '1d')
    ema_34 = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Donchian channels (20-period)
    high_max_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume spike: current volume > 2.0 x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 2.0 * vol_ma_20
    
    # Chop regime filter: avoid high chop (>61.8 = range)
    atr_14 = pd.Series(np.maximum.reduce([
        high[1:] - low[1:],
        np.abs(high[1:] - close[:-1]),
        np.abs(low[1:] - close[:-1])
    ])).rolling(window=14, min_periods=14).mean().values
    atr_14 = np.concatenate([[np.nan], atr_14])  # align length
    true_range = np.maximum.reduce([
        high - low,
        np.abs(high - np.concatenate([[np.nan], close[:-1]])),
        np.abs(low - np.concatenate([[np.nan], close[:-1]]))
    ])
    sum_tr_14 = pd.Series(true_range).rolling(window=14, min_periods=14).sum().values
    atr_for_chop = np.concatenate([[np.nan], sum_tr_14[:len(sum_tr_14)-1]])  # shift for alignment
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(sum_tr_14 / (highest_high_14 - lowest_low_14)) / np.log10(14)
    chop = np.concatenate([[np.nan], chop])
    chop_ok = chop < 61.8  # trending regime
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # EMA34 and Donchian need 20+ bars
    
    for i in range(start_idx, n):
        if np.isnan(ema_34_aligned[i]) or np.isnan(high_max_20[i]) or np.isnan(low_min_20[i]) or np.isnan(vol_ma_20[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        trend_up = price > ema_34_aligned[i]
        trend_down = price < ema_34_aligned[i]
        
        if position == 0:
            # Long: Donchian breakout up + volume spike + trend up + chop OK
            if price > high_max_20[i] and vol_spike[i] and trend_up and chop_ok[i]:
                signals[i] = 0.25
                position = 1
            # Short: Donchian breakout down + volume spike + trend down + chop OK
            elif price < low_min_20[i] and vol_spike[i] and trend_down and chop_ok[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price retrace to Donchian midline or trend fails
            mid = (high_max_20[i] + low_min_20[i]) / 2
            if price < mid or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price retrace to midline or trend fails
            mid = (high_max_20[i] + low_min_20[i]) / 2
            if price > mid or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals