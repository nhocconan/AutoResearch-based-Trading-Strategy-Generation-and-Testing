#!/usr/bin/env python3
"""
4h_Camarilla_Pivot_R1S1_Breakout_Volume_Regime
Strategy: 4h Camarilla pivot (R1/S1) breakout with volume confirmation and chop regime filter.
Long: Break above R1 with volume > 1.5x average and chop > 61.8 (range market).
Short: Break below S1 with volume > 1.5x average and chop > 61.8.
Uses 1d Camarilla levels for structure, chop filter to avoid whipsaw in trends.
Target: 20-40 trades/year per symbol (80-160 total over 4 years).
Works in bull/bear via chop regime filter - only trades in ranging markets.
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
    
    # Get daily data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Previous day's OHLC for Camarilla calculation
    prev_close = np.roll(close_1d, 1)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close[0] = close_1d[0]  # first day uses same day
    prev_high[0] = high_1d[0]
    prev_low[0] = low_1d[0]
    
    # Camarilla levels: R1 = close + (high-low)*1.1/12, S1 = close - (high-low)*1.1/12
    range_1d = prev_high - prev_low
    r1 = prev_close + range_1d * 1.1 / 12
    s1 = prev_close - range_1d * 1.1 / 12
    
    # Choppiness Index: measures if market is ranging (high) or trending (low)
    # CHOP > 61.8 = ranging (good for mean reversion/breakout fade)
    # CHOP < 38.2 = trending (avoid breakouts in strong trends)
    tr1 = np.maximum(high_1d - low_1d, np.absolute(high_1d - np.roll(close_1d, 1)))
    tr2 = np.absolute(np.roll(close_1d, 1) - low_1d)
    tr = np.maximum(tr1, tr2)
    tr[0] = high_1d[0] - low_1d[0]  # first day
    
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero
    range_14 = highest_high - lowest_low
    range_14 = np.where(range_14 == 0, 1e-10, range_14)
    
    chop = 100 * np.log10(atr_14 * 14 / range_14) / np.log10(14)
    chop = np.where(np.isnan(chop), 50, chop)  # neutral if undefined
    
    # Align all daily data to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # need enough for chop calculation
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        vol_confirm = volume[i] > 1.5 * vol_ma[i] if not np.isnan(vol_ma[i]) else False
        
        # Chop regime filter: only trade in ranging markets (CHOP > 61.8)
        regime_filter = chop_aligned[i] > 61.8
        
        if position == 0:
            # Long: price breaks above R1 with volume and in ranging market
            if close[i] > r1_aligned[i] and vol_confirm and regime_filter:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume and in ranging market
            elif close[i] < s1_aligned[i] and vol_confirm and regime_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns below R1 or chop drops (trend emerging)
            if close[i] < r1_aligned[i] or chop_aligned[i] < 50:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns above S1 or chop drops (trend emerging)
            if close[i] > s1_aligned[i] or chop_aligned[i] < 50:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_Pivot_R1S1_Breakout_Volume_Regime"
timeframe = "4h"
leverage = 1.0