#!/usr/bin/env python3
"""
6h_MarketRegime_Breakout
Hypothesis: Combines 12h Donchian breakout with regime detection via ADX and volatility regime.
In trending markets (ADX > 25), trade breakouts in direction of trend. In ranging markets (ADX < 20),
fade reversions at Bollinger Bands. Uses volume confirmation to filter false signals.
Designed to work in both bull and bear markets by adapting to regime.
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
    
    # Get 12h data for Donchian channels
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate 12h Donchian channels (20-period)
    upper_channel = np.full(len(high_12h), np.nan)
    lower_channel = np.full(len(low_12h), np.nan)
    
    for i in range(20, len(high_12h)):
        upper_channel[i] = np.max(high_12h[i-20:i])
        lower_channel[i] = np.min(low_12h[i-20:i])
    
    # Get 6h data for regime detection
    # Calculate ADX(14)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)
    
    for i in range(1, n):
        high_diff = high[i] - high[i-1]
        low_diff = low[i-1] - low[i]
        
        if high_diff > low_diff and high_diff > 0:
            plus_dm[i] = high_diff
        else:
            plus_dm[i] = 0
            
        if low_diff > high_diff and low_diff > 0:
            minus_dm[i] = low_diff
        else:
            minus_dm[i] = 0
            
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # Smooth with Wilder's smoothing (alpha = 1/period)
    atr = np.zeros(n)
    if n >= 1:
        atr[0] = tr[0]
        for i in range(1, n):
            atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    plus_di = np.full(n, np.nan)
    minus_di = np.full(n, np.nan)
    dx = np.full(n, np.nan)
    adx = np.full(n, np.nan)
    
    if n >= 14:
        # Initial values
        plus_dm_sum = np.sum(plus_dm[1:14])
        minus_dm_sum = np.sum(minus_dm[1:14])
        atr_14 = atr[13]
        
        if atr_14 != 0:
            plus_di[13] = (plus_dm_sum / atr_14) * 100
            minus_di[13] = (minus_dm_sum / atr_14) * 100
            if (plus_di[13] + minus_di[13]) != 0:
                dx[13] = (abs(plus_di[13] - minus_di[13]) / (plus_di[13] + minus_di[13])) * 100
        
        # Rolling calculations
        for i in range(14, n):
            plus_dm_sum = plus_dm_sum - plus_dm[i-13] + plus_dm[i]
            minus_dm_sum = minus_dm_sum - minus_dm[i-13] + minus_dm[i]
            atr_14 = (atr[i-1] * 13 + tr[i]) / 14
            
            if atr_14 != 0:
                plus_di[i] = (plus_dm_sum / atr_14) * 100
                minus_di[i] = (minus_dm_sum / atr_14) * 100
                if (plus_di[i] + minus_di[i]) != 0:
                    dx[i] = (abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])) * 100
        
        # Calculate ADX
        if n >= 27:
            adx[26] = np.mean(dx[14:27])
            for i in range(27, n):
                adx[i] = (adx[i-1] * 13 + dx[i]) / 14
    
    # Bollinger Bands (20, 2)
    bb_middle = np.full(n, np.nan)
    bb_std = np.full(n, np.nan)
    bb_upper = np.full(n, np.nan)
    bb_lower = np.full(n, np.nan)
    
    for i in range(20, n):
        bb_middle[i] = np.mean(close[i-20:i])
        bb_std[i] = np.std(close[i-20:i])
        bb_upper[i] = bb_middle[i] + 2 * bb_std[i]
        bb_lower[i] = bb_middle[i] - 2 * bb_std[i]
    
    # Volume spike: current volume > 1.5 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_spike = volume > (vol_ma * 1.5)
    
    # Align 12h channels to 6h timeframe
    upper_aligned = align_htf_to_ltf(prices, df_12h, upper_channel)
    lower_aligned = align_htf_to_ltf(prices, df_12h, lower_channel)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(27, 20)  # ADX needs 27, BB needs 20
    
    for i in range(start_idx, n):
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or 
            np.isnan(adx[i]) or np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Regime-based entry
            if adx[i] > 25:  # Trending regime
                # Long: break above 12h upper channel with volume spike
                if close[i] > upper_aligned[i] and vol_spike[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: break below 12h lower channel with volume spike
                elif close[i] < lower_aligned[i] and vol_spike[i]:
                    signals[i] = -0.25
                    position = -1
            elif adx[i] < 20:  # Ranging regime
                # Long: price touches lower BB with volume spike
                if close[i] <= bb_lower[i] and vol_spike[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: price touches upper BB with volume spike
                elif close[i] >= bb_upper[i] and vol_spike[i]:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Long exit conditions
            if adx[i] > 25:  # In trend - exit on breakdown
                if close[i] < lower_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # In range - exit at middle BB or opposite BB
                if close[i] >= bb_middle[i] or close[i] >= bb_upper[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
        
        elif position == -1:
            # Short exit conditions
            if adx[i] > 25:  # In trend - exit on breakout
                if close[i] > upper_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:  # In range - exit at middle BB or opposite BB
                if close[i] <= bb_middle[i] or close[i] <= bb_lower[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_MarketRegime_Breakout"
timeframe = "6h"
leverage = 1.0