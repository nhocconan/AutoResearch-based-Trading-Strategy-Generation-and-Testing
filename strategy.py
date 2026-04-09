#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot levels from 1d + volume spike + choppiness regime
# Camarilla levels provide high-probability reversal zones from 1d data
# Volume spike confirms institutional interest at pivot touches
# Choppiness regime filter ensures we mean-revert in range (CHOP>61.8) and trend-follow in trend (CHOP<38.2)
# Works in bull/bear: regime adapts, pivots work in both directions
# Target: 50-150 total trades over 4 years (12-37/year) with discrete sizing 0.25

name = "12h_1d_camarilla_pivot_volume_chop_v1"
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
    
    # Load 1d data ONCE before loop for Camarilla and chop calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d Camarilla levels (based on previous day's OHLC)
    # Camarilla: H4 = C + 1.1*(H-L)/2, L4 = C - 1.1*(H-L)/2
    #          H3 = C + 1.1*(H-L)/4, L3 = C - 1.1*(H-L)/4
    #          H2 = C + 1.1*(H-L)/6, L2 = C - 1.1*(H-L)/6
    #          H1 = C + 1.1*(H-L)/12, L1 = C - 1.1*(H-L)/12
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    camarilla_h4 = np.full(len(df_1d), np.nan)
    camarilla_l4 = np.full(len(df_1d), np.nan)
    camarilla_h3 = np.full(len(df_1d), np.nan)
    camarilla_l3 = np.full(len(df_1d), np.nan)
    camarilla_h2 = np.full(len(df_1d), np.nan)
    camarilla_l2 = np.full(len(df_1d), np.nan)
    camarilla_h1 = np.full(len(df_1d), np.nan)
    camarilla_l1 = np.full(len(df_1d), np.nan)
    
    for i in range(1, len(df_1d)):
        hl_range = high_1d[i-1] - low_1d[i-1]
        if hl_range <= 0:
            camarilla_h4[i] = camarilla_l4[i] = np.nan
            camarilla_h3[i] = camarilla_l3[i] = np.nan
            camarilla_h2[i] = camarilla_l2[i] = np.nan
            camarilla_h1[i] = camarilla_l1[i] = np.nan
        else:
            camarilla_h4[i] = close_1d[i-1] + 1.1 * hl_range / 2
            camarilla_l4[i] = close_1d[i-1] - 1.1 * hl_range / 2
            camarilla_h3[i] = close_1d[i-1] + 1.1 * hl_range / 4
            camarilla_l3[i] = close_1d[i-1] - 1.1 * hl_range / 4
            camarilla_h2[i] = close_1d[i-1] + 1.1 * hl_range / 6
            camarilla_l2[i] = close_1d[i-1] - 1.1 * hl_range / 6
            camarilla_h1[i] = close_1d[i-1] + 1.1 * hl_range / 12
            camarilla_l1[i] = close_1d[i-1] - 1.1 * hl_range / 12
    
    # Align Camarilla levels to 12h timeframe (wait for 1d bar close)
    h4_12h = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    l4_12h = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    h3_12h = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    l3_12h = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    h2_12h = align_htf_to_ltf(prices, df_1d, camarilla_h2)
    l2_12h = align_htf_to_ltf(prices, df_1d, camarilla_l2)
    h1_12h = align_htf_to_ltf(prices, df_1d, camarilla_h1)
    l1_12h = align_htf_to_ltf(prices, df_1d, camarilla_l1)
    
    # Calculate 14-period Choppiness Index on 1d
    chop_1d = np.full(len(df_1d), np.nan)
    atr_1d = np.full(len(df_1d), np.nan)
    
    # Calculate ATR first
    tr_1d = np.maximum(
        high_1d[1:] - low_1d[1:],
        np.maximum(
            np.abs(high_1d[1:] - close_1d[:-1]),
            np.abs(low_1d[1:] - close_1d[:-1])
        )
    )
    tr_1d = np.concatenate([[np.nan], tr_1d])
    
    for i in range(len(df_1d)):
        if i < 14:
            atr_1d[i] = np.nan
        else:
            atr_1d[i] = np.mean(tr_1d[i-13:i+1])
    
    # Calculate Choppiness Index
    for i in range(len(df_1d)):
        if i < 14:
            chop_1d[i] = np.nan
        else:
            atr_sum = np.sum(tr_1d[i-13:i+1])
            hh = np.max(high_1d[i-13:i+1])
            ll = np.min(low_1d[i-13:i+1])
            if hh - ll == 0:
                chop_1d[i] = 100
            else:
                chop_1d[i] = 100 * np.log10(atr_sum / np.log10(hh - ll)) / np.log10(14)
    
    # Align Choppiness Index to 12h timeframe
    chop_12h = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Calculate 20-period average volume for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(n):
        if i < 20:
            avg_volume[i] = np.nan
        else:
            avg_volume[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(h4_12h[i]) or np.isnan(l4_12h[i]) or
            np.isnan(h3_12h[i]) or np.isnan(l3_12h[i]) or
            np.isnan(chop_12h[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 2.0x 20-period average
        volume_confirmed = volume[i] > 2.0 * avg_volume[i]
        
        if position == 1:  # Long position
            # Exit: price < L3 (strong support broken) OR chop < 38.2 (trending regime) AND price < H1 (failed to hold)
            if close[i] < l3_12h[i] or (chop_12h[i] < 38.2 and close[i] < h1_12h[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price > H3 (strong resistance broken) OR chop < 38.2 (trending regime) AND price > L1 (failed to hold)
            if close[i] > h3_12h[i] or (chop_12h[i] < 38.2 and close[i] > l1_12h[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic with volume confirmation and Camarilla levels + chop regime
            if volume_confirmed:
                # Determine regime: chop > 61.8 = ranging (mean revert), chop < 38.2 = trending (trend follow)
                if chop_12h[i] > 61.8:  # Ranging regime - mean revert at strong levels
                    # Long entry: price < L4 AND price > L3 (bounce from strong support)
                    if close[i] < l4_12h[i] and close[i] > l3_12h[i]:
                        position = 1
                        signals[i] = 0.25
                    # Short entry: price > H4 AND price < H3 (rejection from strong resistance)
                    elif close[i] > h4_12h[i] and close[i] < h3_12h[i]:
                        position = -1
                        signals[i] = -0.25
                elif chop_12h[i] < 38.2:  # Trending regime - follow trend from median levels
                    # Long entry: price > H3 AND price < H4 (breakout with pullback to median resistance)
                    if close[i] > h3_12h[i] and close[i] < h4_12h[i]:
                        position = 1
                        signals[i] = 0.25
                    # Short entry: price < L3 AND price > L4 (breakdown with pullback to median support)
                    elif close[i] < l3_12h[i] and close[i] > l4_12h[i]:
                        position = -1
                        signals[i] = -0.25
    
    return signals