#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Donchian channel breakout with 1-day Choppiness index regime filter and volume confirmation.
# Long when: Close > Upper Donchian(20) AND Choppiness > 61.8 (range) AND volume > 1.5 * EMA20(volume).
# Short when: Close < Lower Donchian(20) AND Choppiness > 61.8 (range) AND volume > 1.5 * EMA20(volume).
# Uses Donchian for price channels, Choppiness to identify ranging markets for mean reversion,
# volume for momentum confirmation. Designed for low trade frequency (~20-30/year) to minimize fee drag.
# Works in bull/bear via mean reversion in ranging markets identified by Choppiness.
name = "12h_Donchian_Chop_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian Channel: 20-period high/low
    highest_20 = np.full(n, np.nan)
    lowest_20 = np.full(n, np.nan)
    for i in range(20, n):
        highest_20[i] = np.max(high[i-20:i])
        lowest_20[i] = np.min(low[i-20:i])
    
    # Load 1d data for Choppiness Index
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range for Choppiness
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Choppiness Index: 100 * log10(sum(TR) / (ATR * n)) / log10(n)
    atr_14_1d = np.full(len(df_1d), np.nan)
    for i in range(14, len(df_1d)):
        atr_14_1d[i] = np.mean(tr_1d[i-14:i])
    
    chop = np.full(len(df_1d), np.nan)
    lookback = 14
    for i in range(lookback, len(df_1d)):
        sum_tr = np.sum(tr_1d[i-lookback:i])
        atr_val = atr_14_1d[i]
        if not np.isnan(atr_val) and atr_val > 0:
            chop[i] = 100 * np.log10(sum_tr / (atr_val * lookback)) / np.log10(lookback)
    
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Volume confirmation: current volume > 1.5 * 20-period EMA of volume
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Sufficient warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or 
            np.isnan(chop_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Close > Upper Donchian AND Chop > 61.8 (range) AND volume spike
            long_condition = (close[i] > highest_20[i]) and (chop_aligned[i] > 61.8) and volume_spike[i]
            # Short: Close < Lower Donchian AND Chop > 61.8 (range) AND volume spike
            short_condition = (close[i] < lowest_20[i]) and (chop_aligned[i] > 61.8) and volume_spike[i]
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Close < Lower Donchian or Chop drops below 38.2 (trend)
            if close[i] < lowest_20[i] or chop_aligned[i] < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Close > Upper Donchian or Chop drops below 38.2 (trend)
            if close[i] > highest_20[i] or chop_aligned[i] < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals