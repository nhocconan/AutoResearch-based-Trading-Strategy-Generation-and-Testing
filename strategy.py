#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1w EMA50 trend filter and volume spike
# Long when price breaks above Donchian upper band AND price > 1w EMA50 AND volume > 2.0x 20-bar avg
# Short when price breaks below Donchian lower band AND price < 1w EMA50 AND volume > 2.0x 20-bar avg
# Exit when price retests Donchian middle band (20-bar midpoint)
# Uses discrete position sizing (0.25) to reduce fee drag and improve test generalization.
# Target: 20-40 trades/year on 4h timeframe (80-160 total over 4 years) to avoid overtrading.
# Focuses on stronger breakouts with weekly trend alignment and volume confirmation
# to capture high-probability moves while minimizing false signals in choppy markets.
# Works in bull markets by capturing breakouts and in bear markets by shorting breakdowns
# with trend alignment preventing counter-trend trades.

name = "4h_Donchian20_1wEMA50_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA50 for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Donchian(20) channels
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    middle_band = (highest_high + lowest_low) / 2.0
    
    # Volume confirmation: >2.0x 20-bar average volume (stricter to reduce trades)
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(lookback, 50, 20)  # Donchian, EMA50, and volume MA warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(middle_band[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        curr_ema50_1w = ema_50_1w_aligned[i]
        curr_highest = highest_high[i]
        curr_lowest = lowest_low[i]
        curr_middle = middle_band[i]
        curr_close = close[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: price retests Donchian middle band
            if curr_close <= curr_middle:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price retests Donchian middle band
            if curr_close >= curr_middle:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long when price breaks above Donchian upper band AND price > 1w EMA50 AND volume confirmation
            if curr_close > curr_highest and curr_close > curr_ema50_1w and vol_conf:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below Donchian lower band AND price < 1w EMA50 AND volume confirmation
            elif curr_close < curr_lowest and curr_close < curr_ema50_1w and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals