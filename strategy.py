#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot levels (from 1d) + volume spike + chop regime filter
# Uses 1d Camarilla levels (L3, L4, H3, H4) as institutional support/resistance
# Volume confirmation: current volume > 2.0x 24-period average (12h * 2 = 1d)
# Chop regime: only trade when CHOP(14) > 61.8 (ranging market) for mean reversion
# Works in bull/bear: mean reversion at pivots effective in both trends and ranges
# Target: 50-150 total trades over 4 years (12-37/year) with discrete sizing 0.25-0.30

name = "12h_1d_camarilla_volume_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for Camarilla levels and chop regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d Camarilla levels (based on previous day's range)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_h4 = np.full(len(df_1d), np.nan)
    camarilla_h3 = np.full(len(df_1d), np.nan)
    camarilla_l3 = np.full(len(df_1d), np.nan)
    camarilla_l4 = np.full(len(df_1d), np.nan)
    
    for i in range(len(df_1d)):
        if i < 1:
            camarilla_h4[i] = np.nan
            camarilla_h3[i] = np.nan
            camarilla_l3[i] = np.nan
            camarilla_l4[i] = np.nan
        else:
            # Previous day's range
            prev_high = high_1d[i-1]
            prev_low = low_1d[i-1]
            prev_close = close_1d[i-1]
            range_val = prev_high - prev_low
            
            # Camarilla levels
            camarilla_h4[i] = prev_close + range_val * 1.1 / 2
            camarilla_h3[i] = prev_close + range_val * 1.1 / 4
            camarilla_l3[i] = prev_close - range_val * 1.1 / 4
            camarilla_l4[i] = prev_close - range_val * 1.1 / 2
    
    # Align Camarilla levels to 12h timeframe
    h4_12h = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    h3_12h = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    l3_12h = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    l4_12h = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # Calculate 1d Chopiness Index (CHOP) for regime filter
    # CHOP > 61.8 = ranging (good for mean reversion), CHOP < 38.2 = trending
    tr_1d = np.maximum(np.maximum(high_1d - low_1d, np.abs(high_1d - np.roll(close_1d, 1))), 
                       np.abs(low_1d - np.roll(close_1d, 1)))
    # Handle first bar
    tr_1d[0] = high_1d[0] - low_1d[0]
    
    atr_14_1d = np.full(len(df_1d), np.nan)
    for i in range(len(df_1d)):
        if i < 14:
            atr_14_1d[i] = np.nan
        else:
            atr_14_1d[i] = np.mean(tr_1d[i-13:i+1])
    
    chop_1d = np.full(len(df_1d), np.nan)
    for i in range(len(df_1d)):
        if i < 14 or np.isnan(atr_14_1d[i]) or atr_14_1d[i] == 0:
            chop_1d[i] = np.nan
        else:
            # Chopiness Index formula: 100 * log10(sum(TR,14) / (ATR(14) * 14)) / log10(14)
            sum_tr_14 = np.sum(tr_1d[i-13:i+1])
            chop_1d[i] = 100 * np.log10(sum_tr_14 / (atr_14_1d[i] * 14)) / np.log10(14)
    
    chop_12h = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Calculate 24-period average volume for volume confirmation (2 days of 12h = 1 day)
    avg_volume = np.full(n, np.nan)
    for i in range(n):
        if i < 24:
            avg_volume[i] = np.nan
        else:
            avg_volume[i] = np.mean(volume[i-24:i])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(h4_12h[i]) or np.isnan(h3_12h[i]) or np.isnan(l3_12h[i]) or np.isnan(l4_12h[i]) or
            np.isnan(chop_12h[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 2.0x 24-period average
        volume_confirmed = volume[i] > 2.0 * avg_volume[i]
        
        # Chop regime filter: only trade in ranging markets (CHOP > 61.8)
        chop_filter = chop_12h[i] > 61.8
        
        if position == 1:  # Long position
            # Exit: price > H3 (take profit) OR price < L4 (stop loss)
            if close[i] > h3_12h[i] or close[i] < l4_12h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price < L3 (take profit) OR price > H4 (stop loss)
            if close[i] < l3_12h[i] or close[i] > h4_12h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic with volume confirmation, chop filter, and Camarilla levels
            if volume_confirmed and chop_filter:
                # Long entry: price < L4 AND price > L3 (mean reversion from strong support)
                if close[i] < l4_12h[i] and close[i] > l3_12h[i]:
                    position = 1
                    signals[i] = 0.25
                # Short entry: price > H4 AND price < H3 (mean reversion from strong resistance)
                elif close[i] > h4_12h[i] and close[i] < h3_12h[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals