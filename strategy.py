#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Weekly Pivot Breakout with Volume Confirmation
# Long when price breaks above weekly R2 with volume spike
# Short when price breaks below weekly S2 with volume spike
# Exit when price returns to weekly pivot level
# Weekly pivots provide institutional support/resistance levels
# Volume confirmation filters false breakouts
# Designed for low trade frequency (~15-25/year) with edge in both bull and bear markets

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data for pivot calculation
    df_w = get_htf_data(prices, '1w')
    high_w = df_w['high'].values
    low_w = df_w['low'].values
    close_w = df_w['close'].values
    
    # Calculate weekly pivot points (using previous week's OHLC)
    # Pivot = (H + L + C) / 3
    # R2 = Pivot + (H - L)
    # S2 = Pivot - (H - L)
    pivot_w = (high_w + low_w + close_w) / 3
    r2_w = pivot_w + (high_w - low_w)
    s2_w = pivot_w - (high_w - low_w)
    
    # Align weekly pivot levels to 6h timeframe
    pivot_w_aligned = align_htf_to_ltf(prices, df_w, pivot_w)
    r2_w_aligned = align_htf_to_ltf(prices, df_w, r2_w)
    s2_w_aligned = align_htf_to_ltf(prices, df_w, s2_w)
    
    # Calculate 20-period average volume for volume spike detection
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(pivot_w_aligned[i]) or 
            np.isnan(r2_w_aligned[i]) or 
            np.isnan(s2_w_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        pivot_val = pivot_w_aligned[i]
        r2_val = r2_w_aligned[i]
        s2_val = s2_w_aligned[i]
        
        # Volume filter: current volume > 1.5 * 20-day average
        vol_spike = vol > 1.5 * vol_ma
        
        if position == 0:
            # Long conditions: price breaks above weekly R2 + volume spike
            if price > r2_val and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below weekly S2 + volume spike
            elif price < s2_val and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: price returns to weekly pivot level
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when price returns to weekly pivot
                if price <= pivot_val:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when price returns to weekly pivot
                if price >= pivot_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_Weekly_Pivot_R2_S2_Breakout_Volume"
timeframe = "6h"
leverage = 1.0