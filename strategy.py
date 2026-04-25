#!/usr/bin/env python3
"""
4h_Camarilla_H3L3_Breakout_1dTrend_VolumeSpike_Regime
Hypothesis: 4h breakout of Camarilla H3/L3 levels (stronger reversal points than R1/S1) with 1d EMA50 trend filter, volume confirmation (>2.0x 20-bar avg), and chop regime filter (CHOP < 38.2 for trending). Uses tighter entry to reduce trades and avoid fee drag. Works in bull/bear by following 1d trend.
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
    
    # 1d data for HTF trend filter and Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Previous day's Camarilla levels (H3, L3, PP)
    pp = (high_1d + low_1d + close_1d) / 3.0
    h3 = pp + (high_1d - low_1d) * 1.1 / 4.0
    l3 = pp - (high_1d - low_1d) * 1.1 / 4.0
    
    # Align Camarilla levels to 4h timeframe (use previous day's levels)
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    
    # Volume spike: current volume > 2.0x 20-period average (stricter)
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
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need EMA50 (50), volume MA (20), chop (14)
    start_idx = max(50, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(chop_14[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        
        if position == 0:
            # Long: price breaks above H3 in 1d uptrend with volume spike and trending market
            close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
            long_signal = (curr_close > h3_aligned[i]) and \
                         (close_1d_aligned[i] > ema_50_1d_aligned[i]) and \
                         volume_spike[i] and \
                         chop_trending[i]
            # Short: price breaks below L3 in 1d downtrend with volume spike and trending market
            short_signal = (curr_close < l3_aligned[i]) and \
                          (close_1d_aligned[i] < ema_50_1d_aligned[i]) and \
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
            # Exit: price breaks below L3 OR trend turns down
            if (curr_close < l3_aligned[i]) or \
               (close_1d_aligned[i] < ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price breaks above H3 OR trend turns up
            if (curr_close > h3_aligned[i]) or \
               (close_1d_aligned[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_H3L3_Breakout_1dTrend_VolumeSpike_Regime"
timeframe = "4h"
leverage = 1.0