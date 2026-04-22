#!/usr/bin/env python3

"""
Hypothesis: 4-hour Donchian breakout with 1-day trend filter and volume confirmation.
Trades breakouts above/below 20-period Donchian channels in the direction of the 1-day EMA trend.
Adds a chop regime filter to avoid false signals in ranging markets.
Targets 20-50 trades/year (80-200 total over 4 years) with strict entry conditions.
Works in both bull and bear markets by filtering with higher timeframe trend and regime.
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
    
    # Load 1d data for trend and chop filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 1d EMA for trend filter (34-period)
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Chop filter: 14-period Choppy Index on 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    atr_1d = np.zeros(len(df_1d))
    tr_1d = np.maximum(
        high_1d[1:] - low_1d[1:],
        np.maximum(
            np.abs(high_1d[1:] - close_1d[:-1]),
            np.abs(low_1d[1:] - close_1d[:-1])
        )
    )
    atr_1d[1:] = tr_1d
    atr_1d[0] = high_1d[0] - low_1d[0]
    
    # Sum of true ranges over 14 periods
    tr_sum_14 = pd.Series(tr_1d).rolling(window=14, min_periods=14).sum().values
    # Chop = 100 * log10(sum(tr14) / (max(high14)-min(low14))) / log10(14)
    max_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    range_14 = max_high_14 - min_low_14
    chop = np.zeros(len(df_1d))
    mask = (tr_sum_14 > 0) & (range_14 > 0)
    chop[mask] = 100 * np.log10(tr_sum_14[mask] / range_14[mask]) / np.log10(14)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Donchian channels on 4h (20-period)
    max_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    min_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(chop_aligned[i]) or
            np.isnan(max_high_20[i]) or np.isnan(min_low_20[i]) or
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend and regime filters
        uptrend = close[i] > ema_34_1d_aligned[i]
        downtrend = close[i] < ema_34_1d_aligned[i]
        ranging = chop_aligned[i] > 61.8  # Chop > 61.8 indicates ranging market
        
        # Volume confirmation
        vol_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        if position == 0 and vol_spike and not ranging:
            # Long: price breaks above 20-period high, in uptrend
            if close[i] > max_high_20[i] and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 20-period low, in downtrend
            elif close[i] < min_low_20[i] and downtrend:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: opposite Donchian break or trend/chop change
            exit_signal = False
            
            if position == 1:
                # Exit long: price breaks below 20-period low OR chop increases (trend weakening)
                if close[i] < min_low_20[i] or chop_aligned[i] > 61.8:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price breaks above 20-period high OR chop increases
                if close[i] > max_high_20[i] or chop_aligned[i] > 61.8:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Donchian_Breakout_1dEMA34_ChopFilter_Volume"
timeframe = "4h"
leverage = 1.0