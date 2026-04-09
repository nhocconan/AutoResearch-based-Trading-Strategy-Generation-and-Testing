#!/usr/bin/env python3
# 12h_daily_pivot_breakout_volume_v2
# Hypothesis: 12h strategy using daily Camarilla pivot levels with volume confirmation and chop regime filter.
# Long when price breaks above daily R4 with volume > 1.5x 20-period average and CHOP > 61.8 (ranging market).
# Short when price breaks below daily S4 with volume > 1.5x 20-period average and CHOP > 61.8.
# Exit when price closes back inside daily R3/S3 levels or CHOP < 38.2 (trending market).
# Uses discrete position sizing (0.25) to minimize fee churn.
# Designed to capture strong breakouts in ranging markets while avoiding false signals in trends.
# Target: 12-37 trades/year (50-150 total over 4 years) on BTC/ETH/SOL.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_daily_pivot_breakout_volume_v2"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Get daily data for pivot levels and chop regime (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # Calculate daily Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    typical_price_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    pivot_1d = typical_price_1d
    r1_1d = close_1d + (range_1d * 1.1 / 12)
    s1_1d = close_1d - (range_1d * 1.1 / 12)
    r2_1d = close_1d + (range_1d * 1.1 / 6)
    s2_1d = close_1d - (range_1d * 1.1 / 6)
    r3_1d = close_1d + (range_1d * 1.1 / 4)
    s3_1d = close_1d - (range_1d * 1.1 / 4)
    r4_1d = close_1d + (range_1d * 1.1 / 2)
    s4_1d = close_1d - (range_1d * 1.1 / 2)
    
    # Calculate daily Chopiness Index (CHOP) for regime filter
    atr_1d = np.zeros_like(close_1d)
    tr_1d = np.maximum(high_1d - low_1d, np.absolute(high_1d - np.roll(close_1d, 1)), np.absolute(low_1d - np.roll(close_1d, 1)))
    tr_1d[0] = high_1d[0] - low_1d[0]
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    
    max_high_1d = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low_1d = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    chop_denominator = np.where((max_high_1d - min_low_1d) != 0, (max_high_1d - min_low_1d), 1e-10)
    chop_raw = np.log10(atr_1d * np.sqrt(14) / chop_denominator) / np.log10(14) * 100
    chop_1d = np.where(np.isnan(chop_raw), 50.0, chop_raw)  # neutral when undefined
    
    # Align all levels and chop to 12h timeframe
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    r2_1d_aligned = align_htf_to_ltf(prices, df_1d, r2_1d)
    s2_1d_aligned = align_htf_to_ltf(prices, df_1d, s2_1d)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(r4_1d_aligned[i]) or np.isnan(s4_1d_aligned[i]) or 
            np.isnan(r3_1d_aligned[i]) or np.isnan(s3_1d_aligned[i]) or
            np.isnan(volume_ma[i]) or np.isnan(chop_1d_aligned[i]) or
            np.isnan(close[i]) or np.isnan(volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        
        # Regime filter: CHOP > 61.8 for ranging market (mean reversion favorable)
        ranging_market = chop_1d_aligned[i] > 61.8
        
        if position == 1:  # Long position
            # Exit: Price closes back below daily R3 (take profit) or chop regime ends
            if close[i] < r3_1d_aligned[i] or chop_1d_aligned[i] < 38.2:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price closes back above daily S3 (take profit) or chop regime ends
            if close[i] > s3_1d_aligned[i] or chop_1d_aligned[i] < 38.2:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Check for breakout with volume confirmation and ranging market
            bullish_breakout = (close[i] > r4_1d_aligned[i]) and volume_confirmed and ranging_market
            bearish_breakout = (close[i] < s4_1d_aligned[i]) and volume_confirmed and ranging_market
            
            if bullish_breakout:
                position = 1
                signals[i] = 0.25
            elif bearish_breakout:
                position = -1
                signals[i] = -0.25
    
    return signals