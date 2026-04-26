#!/usr/bin/env python3
"""
1d_WeeklyDonchian20_Breakout_WeeklyTrend_VolumeSpike_v1
Hypothesis: Weekly Donchian(20) breakout with weekly trend filter and daily volume confirmation. Uses 1d timeframe with 1w HTF for structure. Weekly Donchian captures major breakouts, weekly EMA50 filters trend direction, and daily volume spike confirms institutional interest. Designed for low trade frequency (<25/year) to minimize fee drag while working in both bull and bear markets by only taking trend-aligned breakouts.
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
    
    # Calculate 1w Donchian channels (20-period)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 25:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly Donchian: highest high and lowest low of past 20 weekly candles
    highest_20 = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Align to 1d timeframe (waits for weekly bar close)
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, highest_20)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, lowest_20)
    
    # Weekly trend filter: EMA50 on weekly close
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Daily volume confirmation: volume > 2.0 * 20-day average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_avg)
    
    # Daily ATR for stoploss (14-period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: need enough for all indicators
    start_idx = max(20, 20, 20)  # Donchian, EMA, volume avg
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(volume_spike[i]) or
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        atr_val = atr[i]
        size = 0.25  # 25% position size
        
        if position == 0:
            # Flat - look for breakout with volume confirmation and trend filter
            # Long: break above weekly Donchian high + weekly close > EMA50 + volume spike
            long_entry = (close_val > donchian_high_aligned[i]) and \
                         (close_1w_aligned := close_1w[-1] if len(close_1w) > 0 else 0) > ema_50_1w_aligned[i] and \
                         volume_spike[i]
            # Short: break below weekly Donchian low + weekly close < EMA50 + volume spike
            short_entry = (close_val < donchian_low_aligned[i]) and \
                          (close_1w_aligned := close_1w[-1] if len(close_1w) > 0 else 0) < ema_50_1w_aligned[i] and \
                          volume_spike[i]
            
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
            # Long - exit on weekly Donchian low retracement or ATR stop
            exit_condition = (close_val < donchian_low_aligned[i]) or \
                           (close_val < entry_price - 2.5 * atr_val)
            if exit_condition:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit on weekly Donchian high retracement or ATR stop
            exit_condition = (close_val > donchian_high_aligned[i]) or \
                           (close_val > entry_price + 2.5 * atr_val)
            if exit_condition:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -size
    
    return signals

name = "1d_WeeklyDonchian20_Breakout_WeeklyTrend_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0