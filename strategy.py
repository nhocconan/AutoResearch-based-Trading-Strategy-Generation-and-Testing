#!/usr/bin/env python3
# 4h_1d_camarilla_pivot_v2
# Hypothesis: In both bull and bear markets, price tends to revert to the mean around key intraday support/resistance levels.
# Uses Camarilla pivot levels (based on prior day's OHLC) from 1d timeframe as dynamic support/resistance.
# Enters long when price crosses above S1 with volume confirmation and low chop (range-bound market).
# Enters short when price crosses below R1 with volume confirmation and low chop.
# Exits when price reaches the opposite pivot level (R1 for longs, S1 for shorts) or when chop increases (trending regime).
# Uses 1d for pivot levels and regime filter, 4h for entry timing to avoid overtrading.
# Target: 20-40 trades/year to minimize fee drag while capturing mean reversion in range-bound markets.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_camarilla_pivot_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot levels and chop filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels for each day
    # Based on previous day's OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: 
    # S1 = C - (H-L)*1.05/6
    # S2 = C - (H-L)*1.05/4
    # S3 = C - (H-L)*1.05/2
    # R1 = C + (H-L)*1.05/6
    # R2 = C + (H-L)*1.05/4
    # R3 = C + (H-L)*1.05/2
    # We'll use S1 and R1 for entries, S3 and R3 for exits
    
    # Calculate for each day
    camarilla_s1 = np.zeros(len(close_1d))
    camarilla_r1 = np.zeros(len(close_1d))
    camarilla_s3 = np.zeros(len(close_1d))
    camarilla_r3 = np.zeros(len(close_1d))
    
    for i in range(1, len(close_1d)):  # Start from 1 since we need previous day
        h = high_1d[i-1]
        l = low_1d[i-1]
        c = close_1d[i-1]
        rang = h - l
        
        camarilla_s1[i] = c - (rang * 1.05 / 6)
        camarilla_r1[i] = c + (rang * 1.05 / 6)
        camarilla_s3[i] = c - (rang * 1.05 / 2)
        camarilla_r3[i] = c + (rang * 1.05 / 2)
    
    # For the first day, we don't have previous day data, so set to NaN
    camarilla_s1[0] = np.nan
    camarilla_r1[0] = np.nan
    camarilla_s3[0] = np.nan
    camarilla_r3[0] = np.nan
    
    # Align Camarilla levels to 4h timeframe
    s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    
    # Calculate Choppiness Index on 1d timeframe for regime filter
    # CHOP > 61.8 = ranging (good for mean reversion)
    # CHOP < 38.2 = trending (avoid)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    # Add first TR (for first period) as high-low since no previous close
    tr = np.concatenate([[high_1d[0] - low_1d[0]], tr])
    
    # ATR(14) - smoothed TR
    atr_period = 14
    atr = np.zeros(len(tr))
    atr[:atr_period] = np.nan
    if len(tr) > atr_period:
        atr[atr_period] = np.mean(tr[:atr_period+1])
        for i in range(atr_period+1, len(tr)):
            atr[i] = (atr[i-1] * (atr_period-1) + tr[i]) / atr_period
    
    # Sum of TRUE RANGE over period
    sum_tr = np.zeros(len(tr))
    for i in range(len(tr)):
        if i < atr_period:
            sum_tr[i] = np.nan
        else:
            sum_tr[i] = np.sum(tr[i-atr_period+1:i+1])
    
    # Choppiness Index
    chop = np.zeros(len(tr))
    for i in range(len(tr)):
        if np.isnan(sum_tr[i]) or np.isnan(atr[i]) or atr[i] == 0:
            chop[i] = np.nan
        else:
            chop[i] = 100 * np.log10(sum_tr[i] / (atr[i] * atr_period)) / np.log10(atr_period)
    
    # Align chop to 4h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Volume confirmation: volume > 20-period average
    vol_ma = np.zeros(n)
    for i in range(n):
        if i < 20:
            vol_ma[i] = np.nan
        else:
            vol_ma[i] = np.mean(volume[i-20+1:i+1])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start from index 50 to ensure we have enough data for all indicators
    for i in range(50, n):
        # Skip if any key value is NaN
        if (np.isnan(s1_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(r3_aligned[i]) or
            np.isnan(chop_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        if position == 1:  # Long position
            # Exit conditions:
            # 1. Price reaches R3 (target)
            # 2. Chop drops below 40 (trending regime - avoid mean reversion in trend)
            # 3. Volume drops significantly (loss of momentum)
            if (close[i] >= r3_aligned[i] or 
                chop_aligned[i] < 40 or 
                vol_ratio < 0.5):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions:
            # 1. Price reaches S3 (target)
            # 2. Chop drops below 40 (trending regime - avoid mean reversion in trend)
            # 3. Volume drops significantly (loss of momentum)
            if (close[i] <= s3_aligned[i] or 
                chop_aligned[i] < 40 or 
                vol_ratio < 0.5):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat - look for entry
            # Long entry: price crosses above S1 with volume confirmation in ranging market
            # Short entry: price crosses below R1 with volume confirmation in ranging market
            if (chop_aligned[i] > 61.8 and  # Ranging market
                vol_ratio > 1.5):  # Volume confirmation
                
                # Long: price crosses above S1
                if (close[i] > s1_aligned[i] and 
                    close[i-1] <= s1_aligned[i-1]):
                    position = 1
                    signals[i] = 0.25
                    
                # Short: price crosses below R1
                elif (close[i] < r1_aligned[i] and 
                      close[i-1] >= r1_aligned[i-1]):
                    position = -1
                    signals[i] = -0.25
    
    return signals