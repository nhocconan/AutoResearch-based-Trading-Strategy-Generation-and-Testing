#!/usr/bin/env python3
"""
12h Camarilla R1S1 Breakout + 1d Volume Spike + Chop Regime Filter
Hypothesis: Camarilla R1/S1 levels on 1d act as intraday support/resistance.
Break above R1 with volume spike and choppy market (CHOP>61.8) signals mean reversion long.
Break below S1 with volume spike and choppy market signals mean reversion short.
Uses 12h timeframe for lower trade frequency. Chop filter ensures we only mean revert in ranging markets.
Works in bull/bear via regime adaptation. Volume spike confirms participation.
Target: 12-37 trades/year.
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
    
    # Get 1d data for Camarilla pivot calculation and chop regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need sufficient data for pivot and chop
        return np.zeros(n)
    
    # Calculate 1d Camarilla pivot levels (based on previous day's OHLC)
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Calculate pivot levels using previous day's data
    range_hl = prev_high - prev_low
    camarilla_r1 = prev_close + (range_hl * 1.1 / 12)
    camarilla_s1 = prev_close - (range_hl * 1.1 / 12)
    
    # Align Camarilla levels to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Calculate 1d Choppiness Index (CHOP) for regime filter
    if len(df_1d) >= 14:
        # True Range
        tr1 = pd.Series(df_1d['high']).diff().abs()
        tr2 = (pd.Series(df_1d['high']) - pd.Series(df_1d['close']).shift()).abs()
        tr3 = (pd.Series(df_1d['low']) - pd.Series(df_1d['close']).shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr_sum = tr.rolling(window=14, min_periods=14).sum().values
        
        # Highest high and lowest low over 14 periods
        hh = pd.Series(df_1d['high']).rolling(window=14, min_periods=14).max().values
        ll = pd.Series(df_1d['low']).rolling(window=14, min_periods=14).min().values
        
        # Chop = 100 * log10(atr_sum / (hh - ll)) / log10(14)
        # Avoid division by zero
        hl_range = hh - ll
        chop = np.where(
            (hl_range > 0) & (atr_sum > 0),
            100 * np.log10(atr_sum / hl_range) / np.log10(14),
            50.0  # default to neutral chop
        )
        chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    else:
        chop_aligned = np.full(n, 50.0)  # default to neutral chop
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for data to propagate
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(chop_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        r1_level = r1_aligned[i]
        s1_level = s1_aligned[i]
        chop_value = chop_aligned[i]
        
        # Volume spike: current volume > 2.0 * 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-19:i+1])
        else:
            vol_ma_20 = np.mean(volume[:i+1])
        volume_spike = curr_volume > 2.0 * vol_ma_20
        
        # Chop filter: CHOP > 61.8 indicates ranging market (mean reversion regime)
        chopping_market = chop_value > 61.8
        
        if position == 0:
            # Long: price breaks above R1 AND volume spike AND chopping market
            long_condition = (curr_close > r1_level) and volume_spike and chopping_market
            # Short: price breaks below S1 AND volume spike AND chopping market
            short_condition = (curr_close < s1_level) and volume_spike and chopping_market
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
        elif position == 1:
            # Exit long: price returns below midpoint or chop ends
            midpoint = (r1_level + s1_level) / 2
            if curr_close <= midpoint or chop_value <= 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns above midpoint or chop ends
            midpoint = (r1_level + s1_level) / 2
            if curr_close >= midpoint or chop_value <= 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1S1_Breakout_1dVolumeSpike_ChopFilter_v1"
timeframe = "12h"
leverage = 1.0