#!/usr/bin/env python3
"""
4h Williams Alligator Breakout with 1d EMA50 Trend Filter and Volume Spike
Hypothesis: Williams Alligator (Jaw/Teeth/Lips) identifies trending vs ranging markets. 
Breakouts above/below the Alligator's lips with volume confirmation and aligned with 1d EMA50 trend 
capture strong directional moves. ATR trailing stop manages risk. Designed to work in both bull and bear regimes
by only taking trend-aligned entries.
Target: 20-35 trades/year on 4h (80-140 total over 4 years) to minimize fee drag.
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
    
    # Williams Alligator: SMAs of median price (typical price)
    # Jaw: 13-period SMMA, Teeth: 8-period SMMA, Lips: 5-period SMMA
    # SMMA = smoothed moving average (similar to EMA but with different smoothing)
    median_price = (high + low) / 2.0
    
    # Calculate SMMA using EMA with alpha = 1/period (approximation)
    jaw = pd.Series(median_price).ewm(alpha=1/13, adjust=False, min_periods=13).mean().values
    teeth = pd.Series(median_price).ewm(alpha=1/8, adjust=False, min_periods=8).mean().values
    lips = pd.Series(median_price).ewm(alpha=1/5, adjust=False, min_periods=5).mean().values
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    # ATR for volatility filter and trailing stop
    tr = np.maximum(np.maximum(high - low, np.abs(high - np.roll(close, 1))), np.abs(low - np.roll(close, 1)))
    tr[0] = 0  # first period has no previous close
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 1d EMA50 trend filter (MTF)
    df_1d = get_htf_data(prices, '1d')
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Start index: need enough for all indicators
    start_idx = max(20, 13, 50) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(lips[i]) or np.isnan(teeth[i]) or np.isnan(jaw[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(atr_14[i]) or 
            np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        vol_spike = volume_spike[i]
        
        # Alligator alignment: Lips above Teeth above Jaw = bullish, Lips below Teeth below Jaw = bearish
        alligator_bullish = lips[i] > teeth[i] and teeth[i] > jaw[i]
        alligator_bearish = lips[i] < teeth[i] and teeth[i] < jaw[i]
        
        # Breakout conditions: price breaks above/below Alligator lips with confirmation
        breakout_long = curr_close > lips[i]
        breakout_short = curr_close < lips[i]
        
        if position == 0:
            # Look for entry signals - require: Alligator alignment + breakout + volume spike + 1d EMA trend alignment
            long_entry = alligator_bullish and breakout_long and vol_spike and (curr_close > ema_50_1d_aligned[i])
            short_entry = alligator_bearish and breakout_short and vol_spike and (curr_close < ema_50_1d_aligned[i])
            
            if long_entry:
                signals[i] = 0.25
                position = 1
                highest_since_entry = curr_high
            elif short_entry:
                signals[i] = -0.25
                position = -1
                lowest_since_entry = curr_low
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management: ATR trailing stop
            highest_since_entry = max(highest_since_entry, curr_high)
            exit_level = highest_since_entry - (2.5 * atr_14[i])
            
            if curr_close < exit_level:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management: ATR trailing stop
            lowest_since_entry = min(lowest_since_entry, curr_low)
            exit_level = lowest_since_entry + (2.5 * atr_14[i])
            
            if curr_close > exit_level:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Williams_Alligator_Breakout_1dEMA50_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0