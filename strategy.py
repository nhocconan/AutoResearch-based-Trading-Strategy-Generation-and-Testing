#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot levels from 1d + volume spike + choppiness regime filter
# Uses 1d OHLC to calculate Camarilla levels (H3/L3 for entries, H4/L4 for stops)
# Entry on 12h close touching H3/L3 with volume > 2x 20-period average
# Choppiness regime: only trade when CHOP(14) > 61.8 (ranging market) for mean reversion
# Designed for 12h timeframe to target 12-37 trades/year (50-150 over 4 years)
# Works in bull/bear: Camarilla provides adaptive support/resistance, chop filter avoids trending markets

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
    
    # Load 1d data ONCE before loop for Camarilla levels and chop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d OHLC for Camarilla
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels (based on previous day's range)
    # H3/L3: entry levels, H4/L4: stop levels
    camarilla_h3 = np.full(len(df_1d), np.nan)
    camarilla_l3 = np.full(len(df_1d), np.nan)
    camarilla_h4 = np.full(len(df_1d), np.nan)
    camarilla_l4 = np.full(len(df_1d), np.nan)
    
    for i in range(1, len(df_1d)):
        high_val = high_1d[i-1]
        low_val = low_1d[i-1]
        close_val = close_1d[i-1]
        range_val = high_val - low_val
        
        camarilla_h3[i] = close_val + range_val * 1.1 / 4
        camarilla_l3[i] = close_val - range_val * 1.1 / 4
        camarilla_h4[i] = close_val + range_val * 1.1 / 2
        camarilla_l4[i] = close_val - range_val * 1.1 / 2
    
    # Align 1d Camarilla levels to 12h timeframe
    h3_12h = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    l3_12h = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    h4_12h = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    l4_12h = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # Calculate 12h Choppiness Index (CHOP) for regime filter
    # CHOP > 61.8 = ranging (good for mean reversion), CHOP < 38.2 = trending
    chop = np.full(n, np.nan)
    for i in range(14, n):
        # True range
        tr1 = high[i] - low[i]
        tr2 = abs(high[i] - close[i-1])
        tr3 = abs(low[i] - close[i-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        
        # Sum of true range over 14 periods
        atr_sum = np.sum(tr[i-13:i+1])
        
        # Highest high and lowest low over 14 periods
        hh = np.max(high[i-13:i+1])
        ll = np.min(low[i-13:i+1])
        
        if hh - ll > 0:
            chop[i] = 100 * np.log10(atr_sum / (hh - ll)) / np.log10(14)
        else:
            chop[i] = 50  # neutral when no range
    
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
        if (np.isnan(h3_12h[i]) or np.isnan(l3_12h[i]) or
            np.isnan(h4_12h[i]) or np.isnan(l4_12h[i]) or
            np.isnan(chop[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 2.0x 20-period average
        volume_confirm = volume[i] > 2.0 * avg_volume[i]
        
        # Regime filter: only trade in ranging market (CHOP > 61.8)
        chop_filter = chop[i] > 61.8
        
        if position == 1:  # Long position
            # Exit: price closes below L3 (profit target) OR above H4 (stop loss)
            if close[i] < l3_12h[i] or close[i] > h4_12h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above H3 (profit target) OR below L4 (stop loss)
            if close[i] > h3_12h[i] or close[i] < l4_12h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic with volume and regime confirmation
            if volume_confirm and chop_filter:
                # Long entry: price touches or crosses below L3 (mean reversion long)
                if close[i] <= l3_12h[i]:
                    position = 1
                    signals[i] = 0.25
                # Short entry: price touches or crosses above H3 (mean reversion short)
                elif close[i] >= h3_12h[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals