#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4H Choppiness Index + Volume Spike + Close Reversion
# - Chop > 61.8 indicates ranging market where price tends to revert to mean (pivot)
# - Volume spike confirms participation at reversal points
# - Buy near S1 when choppy and volume high, sell near R1 when choppy and volume high
# - Works in both bull/bear as ranging behavior occurs in all regimes
# - Target: 20-50 trades/year to avoid fee drag

name = "4h_Chop_Volume_MeanReversion"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for pivot and chop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Previous day's OHLC for pivot
    prev_high_1d = df_1d['high'].shift(1).values
    prev_low_1d = df_1d['low'].shift(1).values
    prev_close_1d = df_1d['close'].shift(1).values
    
    # Daily pivot and ranges
    pivot_1d = (prev_high_1d + prev_low_1d + prev_close_1d) / 3
    range_1d = prev_high_1d - prev_low_1d
    r1_1d = pivot_1d + range_1d * 1.1 / 12
    s1_1d = pivot_1d - range_1d * 1.1 / 12
    
    # Align daily levels to 4h
    pivot_4h = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_4h = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_4h = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # Daily Choppiness Index (14-period)
    atr_period = 14
    tr1 = np.maximum(high - low, np.abs(high - np.roll(close, 1)))
    tr2 = np.maximum(np.abs(low - np.roll(close, 1)), tr1)
    tr1[0] = high[0] - low[0]  # first TR
    tr2[0] = np.abs(low[0] - close[0]) if len(close) > 1 else tr1[0]
    tr = np.maximum(tr1, tr2)
    
    # Smooth TR and ranges
    atr = pd.Series(tr).ewm(span=atr_period, adjust=False).mean().values
    hh = pd.Series(high).rolling(window=atr_period, min_periods=atr_period).max().values
    ll = pd.Series(low).rolling(window=atr_period, min_periods=atr_period).min().values
    
    # Chop calculation
    sum_atr = pd.Series(atr).rolling(window=atr_period, min_periods=atr_period).sum().values
    range_hh_ll = hh - ll
    chop = 100 * np.log10(sum_atr / range_hh_ll) / np.log10(atr_period)
    chop = np.where(range_hh_ll == 0, 50, chop)  # avoid div by zero
    
    chop_4h = align_htf_to_ltf(prices, df_1d, chop)
    
    # Volume filter: spike > 2x 24-period average (24*4h = 4 days)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 24  # Wait for volume MA and chop
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(chop_4h[i]) or np.isnan(r1_4h[i]) or np.isnan(s1_4h[i]) or 
            np.isnan(pivot_4h[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_spike = volume[i] > 2.0 * vol_ma[i]  # Volume confirmation
        chop_high = chop_4h[i] > 61.8  # Ranging market
        
        if position == 0:
            # Long near S1 in choppy market with volume spike
            if (chop_high and vol_spike and 
                close[i] <= s1_4h[i] * 1.005):  # slight buffer for entry
                signals[i] = 0.25
                position = 1
            # Short near R1 in choppy market with volume spike
            elif (chop_high and vol_spike and 
                  close[i] >= r1_4h[i] * 0.995):  # slight buffer for entry
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price reaches pivot or chop breaks down (trend emerging)
            if close[i] >= pivot_4h[i] or chop_4h[i] < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price reaches pivot or chop breaks down
            if close[i] <= pivot_4h[i] or chop_4h[i] < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals