#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_Volume_Regime
Hypothesis: Trade Camarilla pivot breakouts on 12h with volume confirmation and 1d choppiness regime filter.
The strategy goes long when price breaks above R1 and short when breaks below S1, only when 1d choppiness > 61.8 (ranging market) and volume > 1.5x 24-period average.
Camarilla levels are derived from the previous 1d OHLC. This works in both bull and bear markets because it targets mean reversion in ranging conditions.
Uses tight entry conditions to target ~20-30 trades/year, avoiding fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla calculation and choppiness filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels (R1, S1) from previous day's OHLC
    # R1 = C + (H-L) * 1.1/12
    # S1 = C - (H-L) * 1.1/12
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    camarilla_r1 = np.full_like(close_1d, np.nan)
    camarilla_s1 = np.full_like(close_1d, np.nan)
    
    for i in range(1, len(close_1d)):
        # Use previous day's OHLC to calculate today's levels
        prev_close = close_1d[i-1]
        prev_high = high_1d[i-1]
        prev_low = low_1d[i-1]
        range_val = prev_high - prev_low
        
        camarilla_r1[i] = prev_close + range_val * 1.1 / 12
        camarilla_s1[i] = prev_close - range_val * 1.1 / 12
    
    # Align Camarilla levels to 12h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Calculate 1d choppiness index (using previous day's data)
    chop_period = 14
    chop = np.full_like(close_1d, np.nan)
    
    if len(close_1d) >= chop_period + 1:
        # True Range
        tr = np.zeros_like(close_1d)
        tr[0] = high_1d[0] - low_1d[0]
        for i in range(1, len(close_1d)):
            tr[i] = max(
                high_1d[i] - low_1d[i],
                abs(high_1d[i] - close_1d[i-1]),
                abs(low_1d[i] - close_1d[i-1])
            )
        
        # Sum of TR over period
        tr_sum = np.full_like(close_1d, np.nan)
        for i in range(chop_period, len(close_1d)):
            tr_sum[i] = np.sum(tr[i-chop_period+1:i+1])
        
        # Chop = 100 * log10(sum(tr) / (n * (max(high) - min(low)))) / log10(n)
        max_h = np.full_like(close_1d, np.nan)
        min_l = np.full_like(close_1d, np.nan)
        for i in range(chop_period, len(close_1d)):
            max_h[i] = np.max(high_1d[i-chop_period+1:i+1])
            min_l[i] = np.min(low_1d[i-chop_period+1:i+1])
            if tr_sum[i] > 0 and (max_h[i] - min_l[i]) > 0:
                chop[i] = 100 * np.log10(tr_sum[i] / (chop_period * (max_h[i] - min_l[i]))) / np.log10(chop_period)
    
    # Align choppiness to 12h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Volume confirmation: volume > 1.5x 24-period average
    vol_ma = np.full_like(volume, np.nan)
    vol_period = 24
    
    if len(volume) >= vol_period:
        for i in range(vol_period, len(volume)):
            vol_ma[i] = np.mean(volume[i-vol_period:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(vol_period, chop_period) + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(chop_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Regime filter: only trade in ranging markets (chop > 61.8)
        regime_filter = chop_aligned[i] > 61.8
        
        if position == 0:
            # Long: price breaks above R1 + volume + regime
            if close[i] > camarilla_r1_aligned[i] and vol_confirm and regime_filter:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 + volume + regime
            elif close[i] < camarilla_s1_aligned[i] and vol_confirm and regime_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below S1 or chop < 38.2 (trending)
            if close[i] < camarilla_s1_aligned[i] or chop_aligned[i] < 38.2:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above R1 or chop < 38.2 (trending)
            if close[i] > camarilla_r1_aligned[i] or chop_aligned[i] < 38.2:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_Volume_Regime"
timeframe = "12h"
leverage = 1.0