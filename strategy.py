#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R1/S1 breakout with 1d HMA21 trend filter and volume confirmation.
Long when price breaks above Camarilla R1 AND price > 1d HMA21 AND volume > 1.5x 20-period average.
Short when price breaks below Camarilla S1 AND price < 1d HMA21 AND volume > 1.5x 20-period average.
Exit when price reverts to Camarilla pivot (PP) OR ATR trailing stop (1.5*ATR from extreme).
Uses 1d HTF for trend alignment and Camarilla levels from daily.
Target: 100-200 total trades over 4 years (25-50/year) with discrete sizing 0.25.
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
    
    # Calculate 1d HMA21 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 21:  # Need enough for HMA
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    half_len = 21 // 2
    sqrt_len = int(np.sqrt(21))
    
    # HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    def wma(values, window):
        if len(values) < window:
            return np.full_like(values, np.nan)
        weights = np.arange(1, window + 1)
        return np.convolve(values, weights, mode='valid') / weights.sum()
    
    wma_half = wma(close_1d, half_len)
    wma_full = wma(close_1d, 21)
    wma_2n_sub_n = 2 * wma_half - wma_full
    hma_21_1d = wma(wma_2n_sub_n, sqrt_len)
    
    # Align HMA to 4h timeframe
    hma_21_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_21_1d)
    
    # Calculate Camarilla levels from previous day (using daily data)
    high_1d_prev = np.roll(df_1d['high'].values, 1)
    low_1d_prev = np.roll(df_1d['low'].values, 1)
    close_1d_prev = np.roll(df_1d['close'].values, 1)
    
    # Camarilla formula: PP = (H+L+C)/3, Range = H-L
    pp = (high_1d_prev + low_1d_prev + close_1d_prev) / 3.0
    rng = high_1d_prev - low_1d_prev
    # R1 = PP + 1.1 * Range / 4, S1 = PP - 1.1 * Range / 4
    r1 = pp + 1.1 * rng / 4.0
    s1 = pp - 1.1 * rng / 4.0
    
    # Align Camarilla levels to 4h timeframe (use previous day's levels)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    
    # 4h volume average (20-period) for spike filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR(14) for 4h trailing stop calculation
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0.0  # for long trailing stop
    lowest_since_entry = 0.0   # for short trailing stop
    
    # Start from index where all indicators are ready
    start_idx = max(20, 21, 1)  # vol_ma20, hma_21_1d, and +1 for roll
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(hma_21_1d_aligned[i]) or 
            np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(pp_aligned[i]) or
            np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        hma_val = hma_21_1d_aligned[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        pp_val = pp_aligned[i]
        vol_ma_val = vol_ma[i]
        atr_val = atr[i]
        
        if position == 0:
            # Long: price breaks above R1 AND price > 1d HMA21 AND volume spike
            if price > r1_val and price > hma_val and volume[i] > 1.5 * vol_ma_val:
                signals[i] = 0.25
                position = 1
                highest_since_entry = price
            # Short: price breaks below S1 AND price < 1d HMA21 AND volume spike
            elif price < s1_val and price < hma_val and volume[i] > 1.5 * vol_ma_val:
                signals[i] = -0.25
                position = -1
                lowest_since_entry = price
        else:
            # Update highest/lowest since entry for trailing stop
            if position == 1:
                highest_since_entry = max(highest_since_entry, price)
            elif position == -1:
                lowest_since_entry = min(lowest_since_entry, price)
            
            # Exit conditions
            exit_signal = False
            
            # Primary exit: price reverts to pivot point (PP)
            if position == 1 and price < pp_val:
                exit_signal = True
            elif position == -1 and price > pp_val:
                exit_signal = True
            
            # ATR-based trailing stop: 1.5 * ATR from highest/lowest since entry
            if position == 1 and price < highest_since_entry - 1.5 * atr_val:
                exit_signal = True
            elif position == -1 and price > lowest_since_entry + 1.5 * atr_val:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Camarilla_R1S1_Breakout_1dHMA21_Trend_VolumeSpike_PPExit_ATRTrailingStop"
timeframe = "4h"
leverage = 1.0