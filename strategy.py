#!/usr/bin/env python3
name = "4h_Donchian_Breakout_Volume_Trend"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # ===== Donchian(20) Breakout (LTF) =====
    highest_20 = np.full(n, np.nan)
    lowest_20 = np.full(n, np.nan)
    for i in range(20, n):
        highest_20[i] = np.max(high[i-20:i])
        lowest_20[i] = np.min(low[i-20:i])
    
    # ===== 1d Trend (HTF) =====
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema20_1d)
    
    # ===== Volume Spike Filter (LTF) =====
    vol_avg_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_avg_20[i] = np.mean(volume[i-20:i])
    vol_spike = volume > (1.5 * vol_avg_20)
    
    # ===== Choppiness Regime Filter (LTF) =====
    atr_14 = np.full(n, np.nan)
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        atr_14[i] = np.mean(tr[max(0, i-13):i+1]) if i >= 13 else np.mean(tr[:i+1])
    
    highest_20_close = np.full(n, np.nan)
    lowest_20_close = np.full(n, np.nan)
    for i in range(20, n):
        highest_20_close[i] = np.max(close[i-20:i])
        lowest_20_close[i] = np.min(close[i-20:i])
    
    chop = np.full(n, np.nan)
    for i in range(20, n):
        atr_sum = np.sum(tr[i-19:i+1])
        max_high = highest_20_close[i]
        min_low = lowest_20_close[i]
        if max_high > min_low:
            chop[i] = 100 * np.log10(atr_sum / (max_high - min_low)) / np.log10(20)
        else:
            chop[i] = 50
    
    chop_threshold = 61.8
    chop_mask = chop > chop_threshold  # ranging market
    
    # ===== Session Filter: 08-20 UTC =====
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(highest_20[i]) or 
            np.isnan(lowest_20[i]) or
            np.isnan(ema20_1d_aligned[i]) or
            np.isnan(vol_avg_20[i]) or
            np.isnan(chop[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Only trade in ranging markets (chop > 61.8)
        if chop_mask[i]:
            if position == 0:
                # Long: price breaks above Donchian high + volume spike + above 1d EMA20
                if (close[i] > highest_20[i] and
                    vol_spike[i] and
                    close[i] > ema20_1d_aligned[i]):
                    signals[i] = 0.25
                    position = 1
                # Short: price breaks below Donchian low + volume spike + below 1d EMA20
                elif (close[i] < lowest_20[i] and
                      vol_spike[i] and
                      close[i] < ema20_1d_aligned[i]):
                    signals[i] = -0.25
                    position = -1
            elif position == 1:
                # Exit long: price breaks below Donchian low or below 1d EMA20
                if close[i] < lowest_20[i] or close[i] < ema20_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price breaks above Donchian high or above 1d EMA20
                if close[i] > highest_20[i] or close[i] > ema20_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
        else:
            # In trending markets, stay flat
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
    
    return signals