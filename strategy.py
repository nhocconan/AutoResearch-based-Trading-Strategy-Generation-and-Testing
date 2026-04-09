#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot long/short with 1d volume spike and chop regime filter
# Uses 1d Camarilla levels (H3/L3) as structure, enters on retest with volume confirmation
# Choppiness index (1d) filters ranging markets - only trades when CHOP > 61.8 (trending)
# Discrete sizing 0.25 limits fee drag. Works in bull/bear: pivot levels adapt, volume confirms breakouts
# Target: 50-150 total trades over 4 years (12-37/year)

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
    
    # Load 1d data ONCE before loop for Camarilla and chop calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d Camarilla levels (H3, L3, H4, L4)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_h3 = np.full(len(df_1d), np.nan)
    camarilla_l3 = np.full(len(df_1d), np.nan)
    camarilla_h4 = np.full(len(df_1d), np.nan)
    camarilla_l4 = np.full(len(df_1d), np.nan)
    pivot = np.full(len(df_1d), np.nan)
    
    for i in range(len(df_1d)):
        if i < 1:  # Need previous day
            camarilla_h3[i] = np.nan
            camarilla_l3[i] = np.nan
            camarilla_h4[i] = np.nan
            camarilla_l4[i] = np.nan
            pivot[i] = np.nan
        else:
            # Previous day's OHLC
            phigh = high_1d[i-1]
            plow = low_1d[i-1]
            pclose = close_1d[i-1]
            
            pivot[i] = (phigh + plow + pclose) / 3.0
            rang = phigh - plow
            
            camarilla_h3[i] = pivot[i] + rang * 1.1 / 4.0
            camarilla_l3[i] = pivot[i] - rang * 1.1 / 4.0
            camarilla_h4[i] = pivot[i] + rang * 1.1 / 2.0
            camarilla_l4[i] = pivot[i] - rang * 1.1 / 2.0
    
    # Calculate 1d Choppiness Index (CHOP)
    def true_range(high, low, prev_close):
        tr1 = high - low
        tr2 = np.abs(high - prev_close)
        tr3 = np.abs(low - prev_close)
        return np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Previous close for TR calculation
    prev_close_1d = np.append([np.nan], close_1d[:-1])
    tr_1d = true_range(high_1d, low_1d, prev_close_1d)
    
    chop = np.full(len(df_1d), np.nan)
    chop_period = 14
    for i in range(len(df_1d)):
        if i < chop_period:
            chop[i] = np.nan
        else:
            # Sum of true ranges over period
            atr = np.mean(tr_1d[i-chop_period+1:i+1])
            # True range high-low range
            high_low_range = np.max(high_1d[i-chop_period+1:i+1]) - np.min(low_1d[i-chop_period+1:i+1])
            if high_low_range == 0 or atr == 0 or np.isnan(high_low_range) or np.isnan(atr):
                chop[i] = 50.0
            else:
                chop[i] = 100 * np.log10(atr / high_low_range * np.sqrt(chop_period)) / np.log10(chop_period)
    
    # Align 1d indicators to 12h timeframe (wait for 1d bar close)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Calculate 20-period average volume for volume confirmation (12h)
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
            np.isnan(pivot_aligned[i]) or np.isnan(chop_aligned[i]) or
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.8x 20-period average
        volume_confirmed = volume[i] > 1.8 * avg_volume[i]
        
        # Chop regime filter: only trade when trending (CHOP > 61.8)
        trending_regime = chop_aligned[i] > 61.8
        
        if position == 1:  # Long position
            # Exit: price < Camarilla L3 OR chop < 50 (range market) OR price < pivot
            if (close[i] < camarilla_l3_aligned[i] or 
                chop_aligned[i] < 50.0 or 
                close[i] < pivot_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price > Camarilla H3 OR chop < 50 (range market) OR price > pivot
            if (close[i] > camarilla_h3_aligned[i] or 
                chop_aligned[i] < 50.0 or 
                close[i] > pivot_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic with volume confirmation, chop filter, and Camarilla touch
            if volume_confirmed and trending_regime:
                # Long entry: price touches Camarilla L3 from above AND price > pivot (bullish bias)
                if (low[i] <= camarilla_l3_aligned[i] * 1.002 and  # Allow small buffer for touch
                    high[i] >= camarilla_l3_aligned[i] * 0.998 and
                    close[i] > pivot_aligned[i]):
                    position = 1
                    signals[i] = 0.25
                # Short entry: price touches Camarilla H3 from below AND price < pivot (bearish bias)
                elif (high[i] >= camarilla_h3_aligned[i] * 0.998 and  # Allow small buffer for touch
                      low[i] <= camarilla_h3_aligned[i] * 1.002 and
                      close[i] < pivot_aligned[i]):
                    position = -1
                    signals[i] = -0.25
    
    return signals