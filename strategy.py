#!/usr/bin/env python3
name = "4h_Donchian_Breakout_Volume_Trend"
timeframe = "4h"
leverage = 1.0

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
    
    # ===== Donchian(20) Breakout =====
    donchian_high = np.zeros(n)
    donchian_low = np.zeros(n)
    for i in range(20, n):
        donchian_high[i] = np.max(high[i-20:i])
        donchian_low[i] = np.min(low[i-20:i])
    
    # ===== Daily Trend (HTF) =====
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # ===== Volume Spike Filter (1d) =====
    vol_1d = df_1d['volume'].values
    vol_avg_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = vol_1d > (1.8 * vol_avg_1d)
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d.astype(float))
    
    # ===== Choppiness Regime Filter =====
    # Chop > 61.8 = range (mean revert), Chop < 38.2 = trending (trend follow)
    # We want trending for breakouts, so Chop < 38.2
    atr_period = 14
    tr = np.maximum(high[1:] - low[1:], np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])))
    tr = np.concatenate([[np.inf], tr])
    atr = np.zeros(n)
    for i in range(atr_period, n):
        atr[i] = np.mean(tr[i-atr_period+1:i+1])
    
    max_hh = np.zeros(n)
    min_ll = np.zeros(n)
    for i in range(atr_period, n):
        max_hh[i] = np.max(high[i-atr_period+1:i+1])
        min_ll[i] = np.min(low[i-atr_period+1:i+1])
    
    chop = np.zeros(n)
    for i in range(atr_period, n):
        if max_hh[i] > min_ll[i]:
            chop[i] = 100 * np.log10(np.sum(tr[i-atr_period+1:i+1]) / np.log10(atr_period) / (max_hh[i] - min_ll[i]))
        else:
            chop[i] = 50  # neutral
    
    chop_filter = chop < 38.2  # trending regime
    
    # ===== Session Filter: 08-20 UTC =====
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or
            np.isnan(ema34_1d_aligned[i]) or
            np.isnan(vol_spike_1d_aligned[i]) or
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
        
        if position == 0:
            # Long: Price breaks above Donchian high + daily uptrend + volume spike + trending regime
            if (close[i] > donchian_high[i] and
                close[i] > ema34_1d_aligned[i] and
                vol_spike_1d_aligned[i] > 0.5 and
                chop_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low + daily downtrend + volume spike + trending regime
            elif (close[i] < donchian_low[i] and
                  close[i] < ema34_1d_aligned[i] and
                  vol_spike_1d_aligned[i] > 0.5 and
                  chop_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price closes below Donchian median or trend changes
            donchian_mid = (donchian_high[i] + donchian_low[i]) / 2
            if close[i] < donchian_mid or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price closes above Donchian median or trend changes
            donchian_mid = (donchian_high[i] + donchian_low[i]) / 2
            if close[i] > donchian_mid or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals