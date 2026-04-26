#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_VolumeSpike_ChopFilter_v2
Hypothesis: 4h Donchian(20) breakout with volume confirmation and choppiness regime filter.
Long when price breaks above upper Donchian channel AND volume > 2x 20-period average AND chop < 61.8 (trending).
Short when price breaks below lower Donchian channel AND volume > 2x 20-period average AND chop < 61.8.
Uses 1d trend filter: only long in 1d uptrend (close > EMA50), only short in 1d downtrend (close < EMA50).
ATR-based stoploss: exit when price moves against position by 2.5x ATR(20).
Discrete sizing: 0.25 to limit fee churn. Target: 75-200 trades over 4 years.
Works in bull (breakouts continuation) and bear (breakdown continuation) via 1d trend filter.
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
    
    # Donchian(20) channels
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma)
    
    # Choppiness Index (14) - range: 0-100, <38.2=trending, >61.8=choppy
    chop_period = 14
    atr_list = []
    for i in range(n):
        if i == 0:
            tr = high[i] - low[i]
        else:
                tr = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        atr_list.append(tr)
    atr = pd.Series(atr_list).rolling(window=chop_period, min_periods=chop_period).mean().values
    max_high = pd.Series(high).rolling(window=chop_period, min_periods=chop_period).max().values
    min_low = pd.Series(low).rolling(window=chop_period, min_periods=chop_period).min().values
    # Avoid division by zero
    true_range_sum = pd.Series(atr_list).rolling(window=chop_period, min_periods=chop_period).sum().values
    max_min_range = max_high - min_low
    chop = np.where(max_min_range != 0, 100 * np.log10(true_range_sum / max_min_range) / np.log10(chop_period), 50)
    
    # 1d trend filter: EMA(50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # ATR(20) for stoploss
    atr_period = 20
    tr_list = []
    for i in range(n):
        if i == 0:
            tr = high[i] - low[i]
        else:
            tr = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        tr_list.append(tr)
    atr_20 = pd.Series(tr_list).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: max of lookback periods
    start_idx = max(lookback, 20, chop_period, atr_period) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(vol_ma[i]) or np.isnan(chop[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(atr_20[i])):
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
        chop_val = chop[i]
        regime_long = close_val > ema_50_1d_aligned[i]  # 1d uptrend
        regime_short = close_val < ema_50_1d_aligned[i]  # 1d downtrend
        
        if position == 0:
            # Long: price > upper Donchian AND volume confirm AND chop < 61.8 (trending) AND 1d uptrend
            long_signal = (close_val > highest_high[i-1]) and vol_conf and (chop_val < 61.8) and regime_long
            
            # Short: price < lower Donchian AND volume confirm AND chop < 61.8 (trending) AND 1d downtrend
            short_signal = (close_val < lowest_low[i-1]) and vol_conf and (chop_val < 61.8) and regime_short
            
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
            # Exit: stoploss hit OR price re-enters Donchian channel
            stoploss_level = entry_price - 2.5 * atr_20[i]
            if close_val < stoploss_level or close_val < highest_high[i-1]:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: stoploss hit OR price re-enters Donchian channel
            stoploss_level = entry_price + 2.5 * atr_20[i]
            if close_val > stoploss_level or close_val > lowest_low[i-1]:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Donchian20_Breakout_VolumeSpike_ChopFilter_v2"
timeframe = "4h"
leverage = 1.0