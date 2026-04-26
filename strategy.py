#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_ATR_Volume_Trend
Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation (ATR ratio > 1.3). Donchian channels provide robust structure; breakouts with volume and trend alignment capture strong momentum. Discrete sizing 0.25 limits trades to ~20-40/year. Works in bull/bear via 1d trend filter.
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
    
    # Load 1d data ONCE before loop for HTF trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    close_1d_series = pd.Series(close_1d)
    ema_50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate ATR(14) for volume regime
    atr_period = 14
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
    
    # Calculate ATR ratio (current ATR / 50-period ATR) for volume regime
    atr_ratio = atr / pd.Series(atr).rolling(window=50, min_periods=50).mean().values
    
    # Calculate Donchian(20) channels from 4h data (using lookback window)
    lookback = 20
    # Upper channel: highest high of last 20 periods (excluding current)
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().shift(1).values
    # Lower channel: lowest low of last 20 periods (excluding current)
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().shift(1).values
    
    # Fixed position size to control trade frequency
    fixed_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need 50 for ATR ratio, 20 for Donchian, plus 1 for shifts
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(atr_ratio[i]) or
            np.isnan(highest_high[i]) or
            np.isnan(lowest_low[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        ema_50_val = ema_50_1d_aligned[i]
        vol_spike = atr_ratio[i] > 1.3  # volume regime
        upper_channel = highest_high[i]
        lower_channel = lowest_low[i]
        size = fixed_size
        
        # Entry conditions: Donchian breakout with volume spike AND aligned with 1d EMA50 trend
        # Long: price breaks above upper channel (bullish breakout)
        # Short: price breaks below lower channel (bearish breakout)
        long_entry = (close_val > upper_channel) and vol_spike and (close_val > ema_50_val)
        short_entry = (close_val < lower_channel) and vol_spike and (close_val < ema_50_val)
        
        if position == 0:
            # Flat - look for entry
            if long_entry:
                signals[i] = size
                position = 1
            elif short_entry:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit when price re-enters Donchian channel or trend reversal
            if (close_val < upper_channel and close_val > lower_channel) or close_val < ema_50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit when price re-enters Donchian channel or trend reversal
            if (close_val > lower_channel and close_val < upper_channel) or close_val > ema_50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Donchian20_Breakout_ATR_Volume_Trend"
timeframe = "4h"
leverage = 1.0