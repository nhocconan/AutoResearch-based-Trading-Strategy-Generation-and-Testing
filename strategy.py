#!/usr/bin/env python3
"""
12h_Donchian20_Breakout_1wTrend_VolumeSpike_ATRStop
Hypothesis: 12h Donchian(20) breakout with 1w EMA34 trend filter and volume confirmation.
Uses discrete position sizing (0.25) to limit fee drag and control drawdown.
Works in bull/bear markets via 1w trend filter: long only when price > 1w EMA34,
short only when price < 1w EMA34. Volume spike confirms breakout strength.
ATR-based stoploss (2x ATR) manages risk. Targets 50-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop for HTF trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate 1w EMA34 for trend filter
    close_1w = df_1w['close'].values
    close_1w_series = pd.Series(close_1w)
    ema_34_1w = close_1w_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate 1w ATR(14) for stoploss
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    tr1 = np.abs(high_1w - low_1w)
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr1[0] = np.nan
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1w = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_14_1w)
    
    # Donchian(20) channels on 12h
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume spike: volume > 1.5x 20-period median volume
    volume_series = pd.Series(volume)
    vol_median_20 = volume_series.rolling(window=20, min_periods=20).median().values
    volume_spike = volume > (1.5 * vol_median_20)
    
    # Fixed position size to control trade frequency and drawdown
    fixed_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: need 34 for 1w EMA, 14 for ATR, 20 for Donchian and volume median
    start_idx = max(34, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1w_aligned[i]) or
            np.isnan(atr_14_1w_aligned[i]) or
            np.isnan(highest_high[i]) or
            np.isnan(lowest_low[i]) or
            np.isnan(vol_median_20[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        ema_34_val = ema_34_1w_aligned[i]
        atr_val = atr_14_1w_aligned[i]
        vol_spike = volume_spike[i]
        size = fixed_size
        
        if position == 0:
            # Flat - look for entry
            # Long breakout: price > Donchian upper + volume spike + uptrend (close > EMA34_1w)
            long_entry = (high[i] > highest_high[i]) and vol_spike and (close_val > ema_34_val)
            # Short breakout: price < Donchian lower + volume spike + downtrend (close < EMA34_1w)
            short_entry = (low[i] < lowest_low[i]) and vol_spike and (close_val < ema_34_val)
            
            if long_entry:
                signals[i] = size
                position = 1
                entry_price = close_val
            elif short_entry:
                signals[i] = -size
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit on trend reversal or ATR stoploss
            stop_price = entry_price - 2.0 * atr_val
            if close_val < ema_34_val or close_val < stop_price:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit on trend reversal or ATR stoploss
            stop_price = entry_price + 2.0 * atr_val
            if close_val > ema_34_val or close_val > stop_price:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -size
    
    return signals

name = "12h_Donchian20_Breakout_1wTrend_VolumeSpike_ATRStop"
timeframe = "12h"
leverage = 1.0