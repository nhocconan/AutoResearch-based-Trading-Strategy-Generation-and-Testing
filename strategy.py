#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot (from 1d) + volume spike + chop regime filter
# Camarilla levels provide high-probability reversal points; volume confirms authenticity
# Chop filter avoids whipsaws in ranging markets; works in bull/bear via mean reversion at extremes
# Target: 75-200 total trades over 4 years (19-50/year) with discrete sizing 0.25

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
    
    # Load 1d data ONCE before loop for Camarilla and chop calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d Camarilla levels (based on previous day's OHLC)
    camarilla_h4 = np.full(n, np.nan)  # Resistance
    camarilla_l4 = np.full(n, np.nan)  # Support
    camarilla_h3 = np.full(n, np.nan)  # Resistance
    camarilla_l3 = np.full(n, np.nan)  # Support
    camarilla_h2 = np.full(n, np.nan)  # Resistance
    camarilla_l2 = np.full(n, np.nan)  # Support
    camarilla_h1 = np.full(n, np.nan)  # Resistance
    camarilla_l1 = np.full(n, np.nan)  # Support
    camarilla_p = np.full(n, np.nan)   # Pivot point
    
    for i in range(n):
        if i < 1:  # Need previous day's data
            camarilla_h4[i] = np.nan
            camarilla_l4[i] = np.nan
            camarilla_h3[i] = np.nan
            camarilla_l3[i] = np.nan
            camarilla_h2[i] = np.nan
            camarilla_l2[i] = np.nan
            camarilla_h1[i] = np.nan
            camarilla_l1[i] = np.nan
            camarilla_p[i] = np.nan
        else:
            # Get previous day's OHLC (1d data is already aligned)
            prev_high = df_1d['high'].values[i-1] if i-1 < len(df_1d) else np.nan
            prev_low = df_1d['low'].values[i-1] if i-1 < len(df_1d) else np.nan
            prev_close = df_1d['close'].values[i-1] if i-1 < len(df_1d) else np.nan
            
            if np.isnan(prev_high) or np.isnan(prev_low) or np.isnan(prev_close):
                camarilla_h4[i] = np.nan
                camarilla_l4[i] = np.nan
                camarilla_h3[i] = np.nan
                camarilla_l3[i] = np.nan
                camarilla_h2[i] = np.nan
                camarilla_l2[i] = np.nan
                camarilla_h1[i] = np.nan
                camarilla_l1[i] = np.nan
                camarilla_p[i] = np.nan
            else:
                # Camarilla formulas
                camarilla_p[i] = (prev_high + prev_low + prev_close) / 3
                range_val = prev_high - prev_low
                camarilla_h4[i] = camarilla_p[i] + (range_val * 1.1 / 2)
                camarilla_l4[i] = camarilla_p[i] - (range_val * 1.1 / 2)
                camarilla_h3[i] = camarilla_p[i] + (range_val * 1.1 / 4)
                camarilla_l3[i] = camarilla_p[i] - (range_val * 1.1 / 4)
                camarilla_h2[i] = camarilla_p[i] + (range_val * 1.1 / 6)
                camarilla_l2[i] = camarilla_p[i] - (range_val * 1.1 / 6)
                camarilla_h1[i] = camarilla_p[i] + (range_val * 1.1 / 12)
                camarilla_l1[i] = camarilla_p[i] - (range_val * 1.1 / 12)
    
    # Align 1d Camarilla levels to 4h timeframe (wait for 1d bar close)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_h2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h2)
    camarilla_l2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l2)
    camarilla_h1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h1)
    camarilla_l1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l1)
    camarilla_p_aligned = align_htf_to_ltf(prices, df_1d, camarilla_p)
    
    # Calculate 14-period chop regime (choppiness index) on 1d
    chop = np.full(n, np.nan)
    atr_1d = np.full(n, np.nan)
    
    # Calculate True Range for 1d
    tr_1d = np.full(len(df_1d), np.nan)
    for i in range(len(df_1d)):
        if i == 0:
            tr_1d[i] = df_1d['high'].values[i] - df_1d['low'].values[i]
        else:
            hl = df_1d['high'].values[i] - df_1d['low'].values[i]
            hc = abs(df_1d['high'].values[i] - df_1d['close'].values[i-1])
            lc = abs(df_1d['low'].values[i] - df_1d['close'].values[i-1])
            tr_1d[i] = max(hl, hc, lc)
    
    # Calculate ATR(14) for 1d
    for i in range(len(df_1d)):
        if i < 14:
            atr_1d[i] = np.nan
        else:
            atr_1d[i] = np.mean(tr_1d[i-14:i])
    
    # Align ATR to 4h
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate chop: 100 * log10(sum(ATR14)/(n * (max(high)-min(low)))) / log10(n)
    lookback = 14
    for i in range(n):
        if i < lookback:
            chop[i] = np.nan
        else:
            # Get highest high and lowest low over lookback period in 4h data
            period_high = np.max(high[i-lookback:i+1])
            period_low = np.min(low[i-lookback:i+1])
            period_range = period_high - period_low
            
            if period_range <= 0 or np.isnan(atr_1d_aligned[i]):
                chop[i] = np.nan
            else:
                sum_atr = np.sum(atr_1d_aligned[i-lookback:i+1])
                chop[i] = 100 * np.log10(sum_atr / (lookback * period_range)) / np.log10(lookback)
    
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
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(chop[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.8x 20-period average
        volume_confirmed = volume[i] > 1.8 * avg_volume[i]
        
        # Chop regime: < 38.2 = trending, > 61.8 = ranging, 38.2-61.8 = transition
        # We use chop > 61.8 for mean reversion (ranging market)
        chop_ranging = chop[i] > 61.8
        
        if position == 1:  # Long position
            # Exit: price < Camarilla L3 OR chop < 38.2 (trending market)
            if close[i] < camarilla_l3_aligned[i] or chop[i] < 38.2:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price > Camarilla H3 OR chop < 38.2 (trending market)
            if close[i] > camarilla_h3_aligned[i] or chop[i] < 38.2:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic: mean reversion at Camarilla H3/L3 in ranging market with volume
            if volume_confirmed and chop_ranging:
                # Long entry: price < Camarilla L3 AND price > Camarilla L4 (deep support)
                if close[i] < camarilla_l3_aligned[i] and close[i] > camarilla_l4_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short entry: price > Camarilla H3 AND price < Camarilla H4 (deep resistance)
                elif close[i] > camarilla_h3_aligned[i] and close[i] < camarilla_h4_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals