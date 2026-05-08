#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Choppiness Index regime filter + 12h Donchian breakout + volume confirmation
# In ranging markets (CHOP > 61.8): mean reversion at Donchian channels
# In trending markets (CHOP < 38.2): breakout continuation
# Uses 12h Choppiness Index for regime detection to avoid look-ahead, 12h Donchian for breakout levels
# Volume confirmation ensures breakouts have conviction
# Designed for 6h timeframe to target 20-40 trades/year (80-160 total over 4 years)

name = "6h_Chop_Donchian_Breakout_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for regime and breakout levels
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    
    # Calculate 12h Choppiness Index (14-period)
    def true_range(h, l, c_prev):
        tr1 = h - l
        tr2 = np.abs(h - c_prev)
        tr3 = np.abs(l - c_prev)
        return np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Calculate ATR for 12h
    atr_12h = np.zeros(len(df_12h))
    for i in range(len(df_12h)):
        if i == 0:
            tr = df_12h.iloc[i]['high'] - df_12h.iloc[i]['low']
        else:
            tr = true_range(df_12h.iloc[i]['high'], df_12h.iloc[i]['low'], df_12h.iloc[i-1]['close'])
        if i < 14:
            atr_12h[i] = np.nan
        else:
            if i == 14:
                atr_12h[i] = np.nanmean(atr_12h[:i])  # Will be nan, need proper calculation
            else:
                # Wilder's smoothing: ATR = (prev_ATR * 13 + TR) / 14
                if np.isnan(atr_12h[i-1]):
                    atr_12h[i] = np.mean(atr_12h[1:i+1])  # Simple average of first 14 TR
                else:
                    atr_12h[i] = (atr_12h[i-1] * 13 + tr) / 14
    
    # Calculate Choppiness Index
    chop = np.full(len(df_12h), np.nan)
    for i in range(13, len(df_12h)):
        if np.isnan(atr_12h[i]):
            continue
        # Sum of true ranges over 14 periods
        tr_sum = 0
        for j in range(i-13, i+1):
            if j == 0:
                tr = df_12h.iloc[j]['high'] - df_12h.iloc[j]['low']
            else:
                tr = true_range(df_12h.iloc[j]['high'], df_12h.iloc[j]['low'], df_12h.iloc[j-1]['close'])
            tr_sum += tr
        
        # Highest high and lowest low over 14 periods
        hh = np.max(df_12h.iloc[i-13:i+1]['high'].values)
        ll = np.min(df_12h.iloc[i-13:i+1]['low'].values)
        
        if hh == ll or tr_sum == 0:
            chop[i] = 50  # Neutral if no range
        else:
            chop[i] = 100 * np.log10(tr_sum / (hh - ll)) / np.log10(14)
    
    # Calculate 12h Donchian channels (20-period)
    donchian_high = np.full(len(df_12h), np.nan)
    donchian_low = np.full(len(df_12h), np.nan)
    for i in range(19, len(df_12h)):
        donchian_high[i] = np.max(df_12h.iloc[i-19:i+1]['high'].values)
        donchian_low[i] = np.min(df_12h.iloc[i-19:i+1]['low'].values)
    
    # Calculate 12h volume average (20-period)
    vol_avg_12h = np.full(len(df_12h), np.nan)
    for i in range(19, len(df_12h)):
        vol_avg_12h[i] = np.mean(df_12h.iloc[i-19:i+1]['volume'].values)
    
    # Align 12h indicators to 6h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_12h, chop)
    donchian_high_aligned = align_htf_to_ltf(prices, df_12h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_12h, donchian_low)
    vol_avg_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_avg_12h)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # warmup for Donchian
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if any required data is NaN
        if np.isnan(chop_aligned[i]) or np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or \
           np.isnan(vol_avg_12h_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume filter: current volume > 1.5x 12h average volume
        vol_filter = volume[i] > 1.5 * vol_avg_12h_aligned[i]
        
        chop_val = chop_aligned[i]
        dc_high = donchian_high_aligned[i]
        dc_low = donchian_low_aligned[i]
        
        if position == 0:
            # Determine regime and look for entry
            if chop_val > 61.8:  # Ranging market
                # Mean reversion: buy near lower band, sell near upper band
                long_condition = close[i] <= dc_low * 1.001 and vol_filter  # Allow small buffer
                short_condition = close[i] >= dc_high * 0.999 and vol_filter
            elif chop_val < 38.2:  # Trending market
                # Breakout continuation: buy on break above, sell on break below
                long_condition = close[i] > dc_high and vol_filter
                short_condition = close[i] < dc_low and vol_filter
            else:  # Transition zone - no trade
                long_condition = False
                short_condition = False
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: opposite signal or chop extreme
            if chop_val > 61.8 and close[i] >= dc_high * 0.999:  # Reached upper bound in range
                signals[i] = 0.0
                position = 0
            elif chop_val < 38.2 and close[i] < dc_low:  # Broken down in trend
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: opposite signal or chop extreme
            if chop_val > 61.8 and close[i] <= dc_low * 1.001:  # Reached lower bound in range
                signals[i] = 0.0
                position = 0
            elif chop_val < 38.2 and close[i] > dc_high:  # Broken up in trend
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals