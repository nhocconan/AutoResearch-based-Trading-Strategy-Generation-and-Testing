#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using daily Camarilla pivot levels with volume spike and Choppiness regime filter
# Long when price crosses above Camarilla R3 with volume > 2x average and CHOP > 61.8 (ranging market for mean reversion)
# Short when price crosses below Camarilla S3 with volume > 2x average and CHOP > 61.8
# Exit when price crosses opposite Camarilla level (S1 for long, R1 for short) or CHOP < 38.2 (trending market)
# Daily Camarilla provides intraday support/resistance, volume spike confirms institutional interest,
# Choppiness filter ensures we only trade in ranging markets where mean reversion works.
# Target: 15-35 trades per year (60-140 over 4 years) with 0.25 position sizing.

name = "12h_1dCamarilla_R3S3_Volume_Chop_MeanRev_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate daily Camarilla pivot levels ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Daily OHLC for Camarilla calculation
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # Camarilla levels: H-L range
    range_hl = daily_high - daily_low
    
    # Camarilla resistance levels
    r3 = daily_close + (range_hl * 1.1 / 2)
    r2 = daily_close + (range_hl * 1.1 / 4)
    r1 = daily_close + (range_hl * 1.1 / 6)
    
    # Camarilla support levels
    s1 = daily_close - (range_hl * 1.1 / 6)
    s2 = daily_close - (range_hl * 1.1 / 4)
    s3 = daily_close - (range_hl * 1.1 / 2)
    
    # Align daily Camarilla levels to 12h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Calculate Choppiness Index (14-period) on 12h data
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period TR
    
    # ATR (14-period)
    atr = np.zeros(n)
    atr[13] = np.mean(tr[0:14])
    for i in range(14, n):
        atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # Sum of TRUE ranges over 14 periods
    sum_tr = np.zeros(n)
    for i in range(13, n):
        if i == 13:
            sum_tr[i] = np.sum(tr[0:14])
        else:
            sum_tr[i] = sum_tr[i-1] - tr[i-14] + tr[i]
    
    # Choppiness Index: 100 * log10(sum_tr / (atr * sqrt(14))) / log10(14)
    chop = np.zeros(n)
    for i in range(13, n):
        if atr[i] > 0 and sum_tr[i] > 0:
            chop[i] = 100 * np.log10(sum_tr[i] / (atr[i] * np.sqrt(14))) / np.log10(14)
        else:
            chop[i] = 50  # Neutral value
    
    # Volume confirmation: >2x 50-period average
    vol_ma_50 = np.zeros(n)
    for i in range(49, n):
        if i == 49:
            vol_ma_50[i] = np.mean(volume[0:50])
        else:
            vol_ma_50[i] = vol_ma_50[i-1] + (volume[i] - volume[i-50]) / 50
    
    volume_filter = volume > (2.0 * vol_ma_50)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any critical value is NaN or outside session
        if (np.isnan(r3_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(chop[i]) or np.isnan(volume_filter[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price crosses above Camarilla R3 with volume spike in ranging market
            if close[i] > r3_aligned[i] and volume_filter[i] and chop[i] > 61.8:
                signals[i] = 0.25
                position = 1
            # Short: price crosses below Camarilla S3 with volume spike in ranging market
            elif close[i] < s3_aligned[i] and volume_filter[i] and chop[i] > 61.8:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below Camarilla S1 OR market becomes trending
            if close[i] < s1_aligned[i] or chop[i] < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above Camarilla R1 OR market becomes trending
            if close[i] > r1_aligned[i] or chop[i] < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals