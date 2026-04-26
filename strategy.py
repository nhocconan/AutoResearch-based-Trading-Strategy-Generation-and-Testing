#!/usr/bin/env python3
"""
6h_Williams_VIX_Fix_Confluence_v1
Hypothesis: Williams VIX Fix (WVF) identifies volatility spikes and potential reversals. Combined with 1d EMA50 trend filter and volume confirmation, it captures mean reversion in high volatility regimes. Works in both bull and bear markets by fading extreme WVF readings when aligned with higher timeframe trend and volume spikes.
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
    
    # Williams VIX Fix: measures volatility and identifies market bottoms/tops
    # WVF = (HighestHigh(LowestLow) - Low) / (HighestHigh(LowestLow)) * 100
    # HighestHigh = highest high over lookback period
    # LowestLow = lowest low over lookback period
    lookback = 22
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    # Avoid division by zero
    hl_range = highest_high - lowest_low
    hl_range = np.where(hl_range == 0, 1e-10, hl_range)
    wvf = ((highest_high - low) / hl_range) * 100
    
    # Load 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: volume > 1.5 * 20-period EMA volume
    avg_volume = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (1.5 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = max(lookback, 50, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(wvf[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Discrete position sizing
        base_size = 0.25
        
        # Long logic: WVF > 0.8 (extreme fear/volatility spike) + price > 1d EMA50 (uptrend) + volume spike
        # Fading extreme volatility spikes in uptrend context
        if wvf[i] > 80 and close[i] > ema_50_1d_aligned[i] and volume_spike[i]:
            if position != 1:
                signals[i] = base_size
                position = 1
            else:
                signals[i] = base_size
        # Short logic: WVF > 0.8 (extreme fear/volatility spike) + price < 1d EMA50 (downtrend) + volume spike
        # Fading extreme volatility spikes in downtrend context (for short)
        elif wvf[i] > 80 and close[i] < ema_50_1d_aligned[i] and volume_spike[i]:
            if position != -1:
                signals[i] = -base_size
                position = -1
            else:
                signals[i] = -base_size
        # Exit conditions: WVF normalizes (< 50) or loss of volume confirmation
        elif position == 1 and (wvf[i] < 50 or not volume_spike[i]):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (wvf[i] < 50 or not volume_spike[i]):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
    
    return signals

name = "6h_Williams_VIX_Fix_Confluence_v1"
timeframe = "6h"
leverage = 1.0