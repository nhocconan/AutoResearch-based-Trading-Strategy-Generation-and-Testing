#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot reversal with weekly trend filter and volume confirmation.
# Camarilla levels identify intraday support/resistance with high probability reversal zones.
# Weekly trend filter ensures we trade with the higher timeframe momentum.
# Volume confirmation validates the rejection at pivot levels.
# Designed for 12h timeframe to target 50-150 trades over 4 years (12-37/year).

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Daily data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate weekly EMA(50) for trend filter
    close_1w = df_1w['close'].values
    ema_1w = np.full(len(close_1w), np.nan)
    for i in range(49, len(close_1w)):
        ema_1w[i] = np.mean(close_1w[i-49:i+1])  # Simple MA for efficiency
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Calculate daily Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels for each day
    camarilla_h3 = np.zeros(len(close_1d))
    camarilla_l3 = np.zeros(len(close_1d))
    camarilla_h4 = np.zeros(len(close_1d))
    camarilla_l4 = np.zeros(len(close_1d))
    
    for i in range(len(close_1d)):
        if i == 0:
            camarilla_h3[i] = camarilla_l3[i] = camarilla_h4[i] = camarilla_l4[i] = np.nan
        else:
            # Calculate based on previous day
            ph = high_1d[i-1]
            pl = low_1d[i-1]
            pc = close_1d[i-1]
            rang = ph - pl
            
            camarilla_h3[i] = pc + (rang * 1.1 / 6)
            camarilla_l3[i] = pc - (rang * 1.1 / 6)
            camarilla_h4[i] = pc + (rang * 1.1 / 4)
            camarilla_l4[i] = pc - (rang * 1.1 / 4)
    
    # Align Camarilla levels to 12h timeframe
    h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # Calculate average volume (24-period) for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(23, n):
        avg_volume[i] = np.mean(volume[i-23:i+1])
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):
        # Skip if any required data is not ready
        if (np.isnan(ema_1w_aligned[i]) or 
            np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or
            np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        
        # Volume confirmation: current volume > 1.3x average volume
        volume_confirm = vol > 1.3 * avg_vol
        
        if position == 0:
            # Long: price rejects L3/L4 and closes back above, with volume, in uptrend
            if (price > l3_aligned[i] and 
                close[i-1] <= l3_aligned[i] and  # Was at or below rejection level
                volume_confirm and
                close[i] > ema_1w_aligned[i]):  # Weekly uptrend
                position = 1
                signals[i] = position_size
            # Short: price rejects H3/H4 and closes back below, with volume, in downtrend
            elif (price < h3_aligned[i] and 
                  close[i-1] >= h3_aligned[i] and  # Was at or above rejection level
                  volume_confirm and
                  close[i] < ema_1w_aligned[i]):  # Weekly downtrend
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price reaches H3/H4 or trend changes
            if (price >= h3_aligned[i] or 
                close[i] < ema_1w_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price reaches L3/L4 or trend changes
            if (price <= l3_aligned[i] or 
                close[i] > ema_1w_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1w_Camarilla_Pivot_Reversal_Trend_Filter_v1"
timeframe = "12h"
leverage = 1.0