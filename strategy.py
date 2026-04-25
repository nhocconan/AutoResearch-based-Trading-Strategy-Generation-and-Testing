#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_VolumeSpike_1dTrend_Regime
Hypothesis: 4h Donchian(20) breakout with 1d EMA34 trend filter, volume confirmation (>2.0x 20-bar avg), and chop regime filter (CHOP < 38.2 for trending). Uses ATR-based stoploss to manage risk. Works in bull/bear by following 1d trend. Designed for low trade frequency (<50/year) to minimize fee drag.
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
    
    # 1d data for HTF trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Donchian channels (20-period) on 4h
    def donchian_channels(high_arr, low_arr, period=20):
        upper = np.full_like(high_arr, np.nan)
        lower = np.full_like(low_arr, np.nan)
        for i in range(period-1, len(high_arr)):
            upper[i] = np.max(high_arr[i-period+1:i+1])
            lower[i] = np.min(low_arr[i-period+1:i+1])
        return upper, lower
    
    donchian_upper, donchian_lower = donchian_channels(high, low, 20)
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # Choppiness Index (CHOP) on 14-period: CHOP < 38.2 = trending
    def choppiness_index(high_arr, low_arr, close_arr, period=14):
        tr = np.zeros_like(close_arr)
        atr = np.zeros_like(close_arr)
        for i in range(1, len(close_arr)):
            tr[i] = max(high_arr[i] - low_arr[i], 
                       abs(high_arr[i] - close_arr[i-1]), 
                       abs(low_arr[i] - close_arr[i-1]))
        tr[0] = high_arr[0] - low_arr[0]
        atr[period-1] = np.mean(tr[1:period]) if period > 1 else tr[0]
        for i in range(period, len(tr)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        sum_atr = np.zeros_like(close_arr)
        for i in range(period-1, len(close_arr)):
            sum_atr[i] = np.sum(atr[i-period+1:i+1])
        max_high = np.zeros_like(close_arr)
        min_low = np.zeros_like(close_arr)
        for i in range(period-1, len(close_arr)):
            max_high[i] = np.max(high_arr[i-period+1:i+1])
            min_low[i] = np.min(low_arr[i-period+1:i+1])
        chop = np.full_like(close_arr, 50.0)
        for i in range(period-1, len(close_arr)):
            if max_high[i] != min_low[i]:
                chop[i] = 100 * np.log10(sum_atr[i] / (max_high[i] - min_low[i])) / np.log10(period)
        return chop
    
    chop_14 = choppiness_index(high, low, close, 14)
    chop_trending = chop_14 < 38.2  # Only trade in trending markets
    
    # ATR for stoploss (20-period)
    def atr(high_arr, low_arr, close_arr, period=20):
        tr = np.zeros_like(close_arr)
        atr_vals = np.zeros_like(close_arr)
        for i in range(1, len(close_arr)):
            tr[i] = max(high_arr[i] - low_arr[i], 
                       abs(high_arr[i] - close_arr[i-1]), 
                       abs(low_arr[i] - close_arr[i-1]))
        tr[0] = high_arr[0] - low_arr[0]
        atr_vals[period-1] = np.mean(tr[1:period]) if period > 1 else tr[0]
        for i in range(period, len(tr)):
            atr_vals[i] = (atr_vals[i-1] * (period-1) + tr[i]) / period
        return atr_vals
    
    atr_vals = atr(high, low, close, 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need Donchian (20), EMA34 (34), volume MA (20), chop (14), ATR (20)
    start_idx = max(34, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or
            np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(chop_14[i]) or np.isnan(atr_vals[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        
        if position == 0:
            # Long: price breaks above Donchian upper in 1d uptrend with volume spike and trending market
            close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
            long_signal = (curr_close > donchian_upper[i]) and \
                         (close_1d_aligned[i] > ema_34_1d_aligned[i]) and \
                         volume_spike[i] and \
                         chop_trending[i]
            # Short: price breaks below Donchian lower in 1d downtrend with volume spike and trending market
            short_signal = (curr_close < donchian_lower[i]) and \
                          (close_1d_aligned[i] < ema_34_1d_aligned[i]) and \
                          volume_spike[i] and \
                          chop_trending[i]
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: price breaks below Donchian lower OR trend turns down OR stoploss hit
            if (curr_close < donchian_lower[i]) or \
               (close_1d_aligned[i] < ema_34_1d_aligned[i]) or \
               (curr_close < entry_price - 2.0 * atr_vals[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price breaks above Donchian upper OR trend turns up OR stoploss hit
            if (curr_close > donchian_upper[i]) or \
               (close_1d_aligned[i] > ema_34_1d_aligned[i]) or \
               (curr_close > entry_price + 2.0 * atr_vals[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Donchian20_Breakout_VolumeSpike_1dTrend_Regime"
timeframe = "4h"
leverage = 1.0