#!/usr/bin/env python3
# [24976] 12h_1d_camarilla_volume_reversal_v1
# Hypothesis: 12-hour Camarilla pivot level reversals with daily volume confirmation and choppiness regime filter.
# Long when price touches or breaks below L3 (Camarilla support) and closes back above L3 with volume > 1.5x average and chop > 61.8 (range).
# Short when price touches or breaks above H3 (Camarilla resistance) and closes back below H3 with volume > 1.5x average and chop > 61.8.
# Exit when price reaches opposite H3/L3 level or closes beyond H4/L4 (breakout failure).
# Uses daily Camarilla levels for key support/resistance, effective in ranging markets during both bull and bear cycles.
# Choppiness filter ensures we only trade in ranging conditions where mean reversion works.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_camarilla_volume_reversal_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla levels and choppiness
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily Camarilla levels (based on previous day's OHLC)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: based on previous day's range
    camarilla_h4 = np.full(len(df_1d), np.nan)
    camarilla_h3 = np.full(len(df_1d), np.nan)
    camarilla_l3 = np.full(len(df_1d), np.nan)
    camarilla_l4 = np.full(len(df_1d), np.nan)
    
    for i in range(1, len(df_1d)):
        # Use previous day's data to calculate today's levels (no look-ahead)
        prev_high = high_1d[i-1]
        prev_low = low_1d[i-1]
        prev_close = close_1d[i-1]
        rang = prev_high - prev_low
        
        camarilla_h4[i] = prev_close + 1.5 * rang
        camarilla_h3[i] = prev_close + 1.125 * rang
        camarilla_l3[i] = prev_close - 1.125 * rang
        camarilla_l4[i] = prev_close - 1.5 * rang
    
    # Calculate daily choppiness index (14-period)
    chop = np.full(len(df_1d), np.nan)
    for i in range(14, len(df_1d)):
        # True range
        tr1 = high_1d[i] - low_1d[i]
        tr2 = abs(high_1d[i] - close_1d[i-1])
        tr3 = abs(low_1d[i] - close_1d[i-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        
        # Sum of true ranges over 14 periods
        sum_tr = np.sum(tr[i-13:i+1])  # inclusive
        
        # Highest high and lowest low over 14 periods
        hh = np.max(high_1d[i-13:i+1])
        ll = np.min(low_1d[i-13:i+1])
        
        if hh > ll:
            chop[i] = 100 * np.log10(sum_tr / (hh - ll)) / np.log10(14)
        else:
            chop[i] = 50.0  # neutral when no range
    
    # Align daily Camarilla levels and choppiness to 12-hour timeframe
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Calculate volume moving average (20-period)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after warmup
        # Skip if data not ready
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or
            np.isnan(chop_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        price = close[i]
        
        if position == 1:  # Long
            # Exit: price reaches H3 (profit target) or breaks above H4 (breakout failure)
            if price >= camarilla_h3_aligned[i] or price > camarilla_h4_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short
            # Exit: price reaches L3 (profit target) or breaks below L4 (breakout failure)
            if price <= camarilla_l3_aligned[i] or price < camarilla_l4_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price touches L3 and closes back above with volume confirmation in ranging market
            if (low[i] <= camarilla_l3_aligned[i] and price > camarilla_l3_aligned[i] and
                vol_ratio > 1.5 and chop_aligned[i] > 61.8):
                position = 1
                signals[i] = 0.25
            # Enter short: price touches H3 and closes back below with volume confirmation in ranging market
            elif (high[i] >= camarilla_h3_aligned[i] and price < camarilla_h3_aligned[i] and
                  vol_ratio > 1.5 and chop_aligned[i] > 61.8):
                position = -1
                signals[i] = -0.25
    
    return signals