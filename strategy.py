#!/usr/bin/env python3
"""
4h_Donchian20_VolumeSpike_RegimeFilter_v1
Hypothesis: On 4h timeframe, Donchian(20) breakouts with volume spike confirmation and choppiness regime filter capture strong momentum moves while avoiding whipsaws in ranging markets. Volume spike confirms institutional participation. Choppiness index > 61.8 triggers mean-reversion logic (fade breakouts), < 38.2 triggers trend-following (follow breakouts). This adapts to BTC/ETH bull/bear/range cycles. Target: 75-200 total trades over 4 years (19-50/year).
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
    
    # Load 1d data ONCE before loop for HTF regime filter (choppiness)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d choppiness index (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True range
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])  # align with 1d index
    
    # ATR14
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Choppiness Index = 100 * log10(sum(ATR14) / (max(high)-min(low)) * sqrt(period))
    # We'll calculate rolling max(high)-min(low) over 14 periods
    max_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop_raw = 100 * np.log10(np.nansum(atr_14) / (max_high - min_low) * np.sqrt(14)) if False else np.zeros_like(close_1d)
    # Proper rolling calculation:
    chop = np.full_like(close_1d, np.nan)
    for j in range(14, len(close_1d)):
        atr_sum = np.nansum(tr[j-13:j+1])  # sum of 14 TR values
        hh = np.max(high_1d[j-13:j+1])
        ll = np.min(low_1d[j-13:j+1])
        if hh > ll:
            chop[j] = 100 * np.log10(atr_sum / (hh - ll) * np.sqrt(14))
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Load 1d data ONCE before loop for Donchian channels (20-period)
    df_1d_donch = get_htf_data(prices, '1d')  # reuse or reload - helper is efficient
    
    # Calculate 20-period Donchian channels on 1d
    high_1d = df_1d_donch['high'].values
    low_1d = df_1d_donch['low'].values
    
    # Upper channel: 20-period high
    donch_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    # Lower channel: 20-period low
    donch_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 4h timeframe
    donch_high_aligned = align_htf_to_ltf(prices, df_1d_donch, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_1d_donch, donch_low)
    
    # Volume spike detection on 4h (volume > 2.0x 20-period EMA)
    volume_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (volume_ema * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need sufficient data for all indicators)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donch_high_aligned[i]) or 
            np.isnan(donch_low_aligned[i]) or
            np.isnan(chop_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Regime filter based on choppiness
        chop_value = chop_aligned[i]
        is_trending = chop_value < 38.2  # trend-following regime
        is_choppy = chop_value > 61.8    # mean-reversion regime
        
        # Donchian breakout conditions
        breakout_up = close[i] > donch_high_aligned[i]
        breakout_down = close[i] < donch_low_aligned[i]
        
        # Long logic: 
        # In trending regime: follow breakouts (break above upper channel)
        # In choppy regime: fade breakouts (break below lower channel -> long)
        if ((is_trending and breakout_up) or (is_choppy and breakout_down)) and volume_spike[i]:
            if position != 1:
                signals[i] = 0.25
                position = 1
            else:
                signals[i] = 0.25
        # Short logic:
        # In trending regime: follow breakouts (break below lower channel)
        # In choppy regime: fade breakouts (break above upper channel -> short)
        elif ((is_trending and breakout_down) or (is_choppy and breakout_up)) and volume_spike[i]:
            if position != -1:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = -0.25
        # Exit conditions: 
        # - Trend reversal (choppiness regime shift)
        # - Opposite Donchian touch
        elif position == 1 and ((is_choppy and breakout_up) or (not is_trending and breakout_down)):
            signals[i] = 0.0
            position = 0
        elif position == -1 and ((is_choppy and breakout_down) or (not is_trending and breakout_up)):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_VolumeSpike_RegimeFilter_v1"
timeframe = "4h"
leverage = 1.0