#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with 1d volume confirmation and chop regime filter
# Uses 1d Camarilla levels (H3/L3) from prior day for institutional breakout validation
# 4h price breaks above H3 or below L3 with volume > 1.5x 20-period average
# Chop regime filter: only trade when CHOP(14) > 61.8 (ranging market) for mean reversion at extremes
# Works in bull/bear: fading extremes in ranging markets, avoids strong trends
# Target: 20-50 total trades over 4 years (5-12/year) with discrete sizing 0.25

name = "4h_1d_camarilla_volume_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Calculate 1d Camarilla levels (H3, L3) from prior day's OHLC
    # H3 = Close + 1.1*(High - Low)/2
    # L3 = Close - 1.1*(High - Low)/2
    # We use prior day's levels to avoid look-ahead
    prior_high = np.roll(df_1d['high'].values, 1)
    prior_low = np.roll(df_1d['low'].values, 1)
    prior_close = np.roll(df_1d['close'].values, 1)
    
    # Set first value to nan (no prior day)
    prior_high[0] = np.nan
    prior_low[0] = np.nan
    prior_close[0] = np.nan
    
    camarilla_h3 = prior_close + 1.1 * (prior_high - prior_low) / 2.0
    camarilla_l3 = prior_close - 1.1 * (prior_high - prior_low) / 2.0
    
    # Align 1d Camarilla levels to 4h timeframe (wait for daily close)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Calculate 4h 20-period average volume for confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(n):
        if i < 20:
            avg_volume[i] = np.nan
        else:
            avg_volume[i] = np.mean(volume[i-20:i])
    
    # Calculate 4h Choppiness Index (CHOP) for regime filter
    # CHOP = 100 * log10(sum(ATR(14)) / log10(range(14))) / log10(14)
    # CHOP > 61.8 = ranging market (good for mean reversion)
    # CHOP < 38.2 = trending market (avoid)
    atr_14 = np.full(n, np.nan)
    true_high = np.maximum(high[1:], close[:-1])
    true_low = np.minimum(low[1:], close[:-1])
    true_range = np.maximum(true_high - true_low, 
                           np.maximum(high - close[:-1], close - low[:-1]))
    true_range = np.concatenate([[np.nan], true_range])
    
    for i in range(n):
        if i < 14:
            atr_14[i] = np.nan
        else:
            atr_14[i] = np.mean(true_range[i-13:i+1])
    
    max_high_14 = np.full(n, np.nan)
    min_low_14 = np.full(n, np.nan)
    for i in range(n):
        if i < 14:
            max_high_14[i] = np.nan
            min_low_14[i] = np.nan
        else:
            max_high_14[i] = np.max(high[i-13:i+1])
            min_low_14[i] = np.min(low[i-13:i+1])
    
    chop = np.full(n, np.nan)
    for i in range(n):
        if i < 14 or np.isnan(atr_14[i]) or np.isnan(max_high_14[i]) or np.isnan(min_low_14[i]) or max_high_14[i] == min_low_14[i]:
            chop[i] = np.nan
        else:
            sum_atr = np.sum(atr_14[i-13:i+1])
            range_14 = max_high_14[i] - min_low_14[i]
            chop[i] = 100 * np.log10(sum_atr) / np.log10(14) / np.log10(range_14) if range_14 > 0 else 100
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(avg_volume[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * avg_volume[i]
        
        # Chop regime filter: only trade in ranging markets (CHOP > 61.8)
        chop_filter = chop[i] > 61.8
        
        if position == 1:  # Long position
            # Exit: price < Camarilla L3 (mean reversion complete) OR chop breaks down (trend emerging)
            if close[i] < camarilla_l3_aligned[i] or chop[i] < 50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price > Camarilla H3 (mean reversion complete) OR chop breaks down (trend emerging)
            if close[i] > camarilla_h3_aligned[i] or chop[i] < 50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic with volume confirmation, chop filter, and Camarilla breakout
            if volume_confirmed and chop_filter:
                # Long entry: price < Camarilla L3 (oversold bounce)
                if close[i] < camarilla_l3_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short entry: price > Camarilla H3 (overbought fade)
                elif close[i] > camarilla_h3_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals