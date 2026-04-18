#!/usr/bin/env python3
"""
4h_1d_WickReversal_Volume
Hypothesis: Trade reversals at daily candle wicks in the direction of 4h momentum, confirmed by volume spikes. Long when 4h close > daily low and 4h momentum > 0 with volume > 1.5x average. Short when 4h close < daily high and 4h momentum < 0 with volume > 1.5x average. Works in bull/bear by capturing rejection at daily extremes. Target ~20 trades/year to minimize fee drag.
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
    
    # Get 1d data for daily high/low
    df_1d = get_htf_data(prices, '1d')
    
    # Daily high/low (previous day's values)
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    
    # Previous day's values (completed day)
    prev_daily_high = np.roll(daily_high, 1)
    prev_daily_low = np.roll(daily_low, 1)
    prev_daily_high[0] = daily_high[0]
    prev_daily_low[0] = daily_low[0]
    
    # Align daily levels to 4h timeframe
    daily_high_aligned = align_htf_to_ltf(prices, df_1d, prev_daily_high)
    daily_low_aligned = align_htf_to_ltf(prices, df_1d, prev_daily_low)
    
    # 4h momentum (close - open) - positive = bullish, negative = bearish
    momentum = close - prices['open'].values
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = np.full_like(volume, np.nan)
    vol_period = 20
    if len(volume) >= vol_period:
        for i in range(vol_period, len(volume)):
            vol_ma[i] = np.mean(volume[i - vol_period:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, vol_period)
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(daily_high_aligned[i]) or np.isnan(daily_low_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: 4h close above daily low with bullish momentum and volume
            if close[i] > daily_low_aligned[i] and momentum[i] > 0 and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: 4h close below daily high with bearish momentum and volume
            elif close[i] < daily_high_aligned[i] and momentum[i] < 0 and vol_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: close below daily low or momentum turns bearish
            if close[i] < daily_low_aligned[i] or momentum[i] < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: close above daily high or momentum turns bullish
            if close[i] > daily_high_aligned[i] or momentum[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_1d_WickReversal_Volume"
timeframe = "4h"
leverage = 1.0