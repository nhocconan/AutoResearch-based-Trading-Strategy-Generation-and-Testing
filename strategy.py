#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout + 1d volume spike + choppiness regime filter
# Camarilla levels provide high-probability reversal/breakout points from 1d timeframe
# Volume spike confirms institutional participation at these key levels
# Choppiness filter avoids whipsaws in ranging markets (CHOP > 61.8 = range, < 38.2 = trend)
# Discrete sizing 0.25 limits fee drag while maintaining sufficient exposure
# Works in bull/bear: Camarilla effective in all regimes, volume confirms validity
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe

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
    
    # Calculate 1d Camarilla levels (using previous day's OHLC)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: based on previous day's range
    camarilla_h5 = np.full(len(close_1d), np.nan)  # Resistance 5
    camarilla_h4 = np.full(len(close_1d), np.nan)  # Resistance 4
    camarilla_h3 = np.full(len(close_1d), np.nan)  # Resistance 3
    camarilla_l3 = np.full(len(close_1d), np.nan)  # Support 3
    camarilla_l4 = np.full(len(close_1d), np.nan)  # Support 4
    camarilla_l5 = np.full(len(close_1d), np.nan)  # Support 5
    
    for i in range(1, len(close_1d)):  # Start from 1 to use previous day
        # Previous day's OHLC
        prev_high = high_1d[i-1]
        prev_low = low_1d[i-1]
        prev_close = close_1d[i-1]
        
        # Calculate range
        range_val = prev_high - prev_low
        
        if range_val > 0:
            camarilla_h5[i] = prev_close + 1.1 * range_val * 1.1 / 2
            camarilla_h4[i] = prev_close + 1.1 * range_val * 1.1 / 4
            camarilla_h3[i] = prev_close + 1.1 * range_val * 1.1 / 6
            camarilla_l3[i] = prev_close - 1.1 * range_val * 1.1 / 6
            camarilla_l4[i] = prev_close - 1.1 * range_val * 1.1 / 4
            camarilla_l5[i] = prev_close - 1.1 * range_val * 1.1 / 2
    
    # Align Camarilla levels to 12h timeframe (wait for 1d bar close)
    h5_12h = align_htf_to_ltf(prices, df_1d, camarilla_h5)
    h4_12h = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    h3_12h = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    l3_12h = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    l4_12h = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    l5_12h = align_htf_to_ltf(prices, df_1d, camarilla_l5)
    
    # Calculate 1d Choppiness Index (CHOP) for regime filter
    # CHOP > 61.8 = ranging market (mean revert), CHOP < 38.2 = trending
    high_1d_arr = df_1d['high'].values
    low_1d_arr = df_1d['low'].values
    close_1d_arr = df_1d['close'].values
    
    chop_1d = np.full(len(close_1d_arr), np.nan)
    atr_period = 14
    
    for i in range(atr_period, len(close_1d_arr)):
        # True Range
        tr1 = high_1d_arr[i] - low_1d_arr[i]
        tr2 = abs(high_1d_arr[i] - close_1d_arr[i-1])
        tr3 = abs(low_1d_arr[i] - close_1d_arr[i-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        
        # Sum of TR over period
        atr_sum = np.sum(tr[i-atr_period+1:i+1])
        
        # Highest high and lowest low over period
        hh = np.max(high_1d_arr[i-atr_period+1:i+1])
        ll = np.min(low_1d_arr[i-atr_period+1:i+1])
        
        if hh != ll:
            chop_1d[i] = 100 * np.log10(atr_sum / (hh - ll)) / np.log10(atr_period)
        else:
            chop_1d[i] = 50.0  # Neutral when no range
    
    # Align Choppiness to 12h timeframe
    chop_12h = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Calculate 20-period average volume for volume confirmation (1d)
    avg_volume_1d = np.full(len(close_1d), np.nan)
    for i in range(20, len(close_1d)):
        avg_volume_1d[i] = np.mean(volume[i-20:i])
    
    # Align average volume to 12h timeframe
    avg_volume_12h = align_htf_to_ltf(prices, df_1d, avg_volume_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(h5_12h[i]) or np.isnan(h3_12h[i]) or np.isnan(l3_12h[i]) or
            np.isnan(l5_12h[i]) or np.isnan(chop_12h[i]) or np.isnan(avg_volume_12h[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 2.0x 20-period average
        volume_confirmed = volume[i] > 2.0 * avg_volume_12h[i]
        
        # Chop regime: we prefer trending markets (CHOP < 38.2) for breakouts
        # In ranging markets (CHOP > 61.8), we avoid breakout trades
        chop_filter = chop_12h[i] < 38.2  # Only trade in trending conditions
        
        if position == 1:  # Long position
            # Exit: price < Camarilla L3 OR chop becomes too high (ranging)
            if close[i] < l3_12h[i] or chop_12h[i] > 61.8:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price > Camarilla H3 OR chop becomes too high (ranging)
            if close[i] > h3_12h[i] or chop_12h[i] > 61.8:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic with volume confirmation and chop filter
            if volume_confirmed and chop_filter:
                # Long entry: price > Camarilla H3 (break above resistance)
                if close[i] > h3_12h[i]:
                    position = 1
                    signals[i] = 0.25
                # Short entry: price < Camarilla L3 (break below support)
                elif close[i] < l3_12h[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals