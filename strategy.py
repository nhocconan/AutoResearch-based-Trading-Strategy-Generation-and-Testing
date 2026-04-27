#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for Choppiness Index and Donchian Channel
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 14-period Choppiness Index (CHOP)
    # CHOP = 100 * log10(SUM(ATR1) / (HHV(HIGH,n) - LLV(LOW,n))) / log10(n)
    tr1 = np.maximum(high[1:] - low[1:], np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])))
    tr1 = np.concatenate([[np.nan], tr1])  # align length
    atr1 = pd.Series(tr1).rolling(window=14, min_periods=14).mean().values
    sum_atr1 = pd.Series(atr1).rolling(window=14, min_periods=14).sum().values
    hh14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(sum_atr1 / (hh14 - ll14)) / np.log10(14)
    
    # Calculate 20-period Donchian Channel (upper/lower bands)
    dc_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    dc_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate volume spike (volume > 2.0x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    # Align HTF indicators to 12h timeframe
    chop_12h = align_htf_to_ltf(prices, df_1d, chop)
    dc_upper_12h = align_htf_to_ltf(prices, df_1d, dc_upper)
    dc_lower_12h = align_htf_to_ltf(prices, df_1d, dc_lower)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(chop_12h[i]) or np.isnan(dc_upper_12h[i]) or np.isnan(dc_lower_12h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Long conditions:
        # 1. Choppy market (CHOP > 61.8) AND price breaks above Donchian upper band with volume spike
        long_breakout = (chop_12h[i] > 61.8 and close[i] > dc_upper_12h[i] and volume_spike[i])
        
        # Short conditions:
        # 1. Choppy market (CHOP > 61.8) AND price breaks below Donchian lower band with volume spike
        short_breakout = (chop_12h[i] > 61.8 and close[i] < dc_lower_12h[i] and volume_spike[i])
        
        if long_breakout:
            signals[i] = 0.25
            position = 1
        elif short_breakout:
            signals[i] = -0.25
            position = -1
        # Exit conditions: opposite Donchian breakout with volume confirmation
        elif position == 1 and close[i] < dc_lower_12h[i] and volume_spike[i]:
            signals[i] = 0.0
            position = 0
        elif position == -1 and close[i] > dc_upper_12h[i] and volume_spike[i]:
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_Choppiness_DonchianBreakout_Volume2x_1d"
timeframe = "12h"
leverage = 1.0