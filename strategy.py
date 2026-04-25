#!/usr/bin/env python3
"""
4h Donchian Breakout with Volume Spike and Choppiness Regime Filter
Hypothesis: Donchian(20) breakouts capture strong momentum moves. Volume confirmation
filters weak breakouts, while choppiness regime filter (CHOP > 61.8) ensures we only
trade in ranging markets where mean reversion at channel extremes works well.
Works in both bull and bear markets by trading breakouts with the trend when trending
(CHOP < 38.2) and mean reversion at channel boundaries when ranging (CHOP > 61.8).
4h timeframe targets 20-50 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA34 for trend filter (used as additional confirmation)
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Donchian channels (20-period)
    high_ma = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_ma = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    # Choppiness Index regime filter (14-period)
    def calculate_chop(high_arr, low_arr, close_arr, window=14):
        atr_list = []
        for i in range(len(high_arr)):
            if i == 0:
                tr = high_arr[i] - low_arr[i]
            else:
                tr = max(high_arr[i] - low_arr[i], 
                         abs(high_arr[i] - close_arr[i-1]),
                         abs(low_arr[i] - close_arr[i-1]))
            atr_list.append(tr)
        
        atr_series = pd.Series(atr_list)
        atr_sum = atr_series.rolling(window=window, min_periods=window).sum().values
        
        highest_high = pd.Series(high_arr).rolling(window=window, min_periods=window).max().values
        lowest_low = pd.Series(low_arr).rolling(window=window, min_periods=window).min().values
        range_max_min = highest_high - lowest_low
        
        chop = np.zeros_like(close_arr)
        for i in range(len(close_arr)):
            if atr_sum[i] > 0 and range_max_min[i] > 0:
                chop[i] = 100 * np.log10(atr_sum[i] / range_max_min[i]) / np.log10(window)
            else:
                chop[i] = 50.0
        return chop
    
    chop = calculate_chop(high, low, close, 14)
    chop_range = chop > 61.8  # ranging market
    chop_trend = chop < 38.2  # trending market
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for calculations
    start_idx = max(20, 34)  # Donchian, EMA, volume MA
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_aligned[i]) or np.isnan(high_ma[i]) or 
            np.isnan(low_ma[i]) or np.isnan(vol_ma[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Look for entry signals
            # In trending markets: breakout in direction of trend
            # In ranging markets: mean reversion at channel boundaries
            
            long_entry = False
            short_entry = False
            
            if chop_trend[i]:  # trending market
                # Long: break above upper Donchian AND uptrend (price > 1d EMA34) AND volume spike
                long_entry = (curr_close > high_ma[i]) and (curr_close > ema_34_aligned[i]) and vol_spike
                # Short: break below lower Donchian AND downtrend (price < 1d EMA34) AND volume spike
                short_entry = (curr_close < low_ma[i]) and (curr_close < ema_34_aligned[i]) and vol_spike
            elif chop_range[i]:  # ranging market
                # Long: price near lower Donchian AND volume spike (mean reversion long)
                long_entry = (curr_low <= low_ma[i] * 1.001) and vol_spike and (curr_close > low_ma[i])
                # Short: price near upper Donchian AND volume spike (mean reversion short)
                short_entry = (curr_high >= high_ma[i] * 0.999) and vol_spike and (curr_close < high_ma[i])
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: price breaks below lower Donchian OR loss of momentum
            if curr_close < low_ma[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price breaks above upper Donchian OR loss of momentum
            if curr_close > high_ma[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian_Breakout_VolumeSpike_ChopRegime"
timeframe = "4h"
leverage = 1.0