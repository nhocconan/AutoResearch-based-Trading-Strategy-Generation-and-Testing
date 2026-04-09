#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot levels (H3/L3) from 1d + volume spike (2.0x 20-period avg) + chop regime filter (CHOP > 61.8 = range)
# Camarilla levels provide high-probability reversal points in ranging markets; volume spike confirms institutional interest
# Chop regime filter ensures we only mean-revert in ranging conditions, avoiding trending markets where pivots fail
# Works in bull/bear: chop filter adapts to market regime, volume confirmation avoids false signals
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
    
    # Load 1d data ONCE before loop for Camarilla levels and chop regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d Camarilla levels (H3, L3) using previous day's OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla: H3 = close + 1.1*(high-low)/6, L3 = close - 1.1*(high-low)/6
    camarilla_h3 = close_1d + 1.1 * (high_1d - low_1d) / 6
    camarilla_l3 = close_1d - 1.1 * (high_1d - low_1d) / 6
    
    # Align Camarilla levels to 4h timeframe (completed 1d bar only)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Calculate 1d Choppiness Index (CHOP) for regime filter
    # CHOP = 100 * log10(sum(ATR) / (max(high) - min(low))) / log10(period)
    # We use 14-period CHOP on 1d timeframe
    tr_1d = np.maximum(high_1d - low_1d, 
                       np.maximum(np.abs(high_1d - np.roll(close_1d, 1)), 
                                  np.abs(low_1d - np.roll(close_1d, 1))))
    tr_1d[0] = high_1d[0] - low_1d[0]  # First TR
    atr_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    max_high_1d = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low_1d = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    chop_1d = np.full(len(df_1d), np.nan)
    for i in range(14, len(df_1d)):
        if atr_1d[i] > 0 and (max_high_1d[i] - min_low_1d[i]) > 0:
            chop_1d[i] = 100 * np.log10(atr_1d[i] * 14 / (max_high_1d[i] - min_low_1d[i])) / np.log10(14)
        else:
            chop_1d[i] = 50  # Neutral value when calculation invalid
    
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
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
            np.isnan(chop_1d_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 2.0x 20-period average (strict filter)
        volume_confirmed = volume[i] > 2.0 * avg_volume[i]
        
        # Chop regime filter: only trade when CHOP > 61.8 (strong ranging market)
        chop_filter = chop_1d_aligned[i] > 61.8
        
        if position == 1:  # Long position
            # Exit: price >= Camarilla H3 (profit target) OR chop regime ends
            if close[i] >= camarilla_h3_aligned[i] or chop_1d_aligned[i] <= 61.8:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price <= Camarilla L3 (profit target) OR chop regime ends
            if close[i] <= camarilla_l3_aligned[i] or chop_1d_aligned[i] <= 61.8:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic with volume confirmation, chop regime, and Camarilla touch
            if volume_confirmed and chop_filter:
                # Long entry: price touches or crosses above Camarilla L3 AND below H3 (long bias in range)
                if low[i] <= camarilla_l3_aligned[i] and high[i] < camarilla_h3_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short entry: price touches or crosses below Camarilla H3 AND above L3 (short bias in range)
                elif high[i] >= camarilla_h3_aligned[i] and low[i] > camarilla_l3_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals