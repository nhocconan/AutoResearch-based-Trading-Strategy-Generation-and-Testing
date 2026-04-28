#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d volume spike and 1d choppiness regime filter.
# Enter long when price breaks above Camarilla R3 with volume > 2.0x 20-bar average and CHOP > 61.8 (rangy market).
# Enter short when price breaks below Camarilla S3 with volume > 2.0x 20-bar average and CHOP > 61.8.
# Exit when price reverts to Camarilla Pivot point or opposite breakout occurs.
# Camarilla levels provide intraday support/resistance, volume spike confirms institutional interest,
# and chop filter ensures we only trade in ranging markets where mean reversion works.
# Uses discrete position sizing (0.25) to control risk. Target: 50-150 total trades over 4 years.

name = "12h_Camarilla_R3S3_Breakout_1dVolumeSpike_ChopFilter_v1"
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
    
    # Get 1d data for Camarilla and chop calculation (HTF)
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d Camarilla levels (based on previous day's OHLC)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: based on previous day's range
    # R4 = close + 1.5*(high-low), R3 = close + 1.125*(high-low), etc.
    # S4 = close - 1.5*(high-low), S3 = close - 1.125*(high-low)
    range_1d = high_1d - low_1d
    camarilla_pivot = (high_1d + low_1d + close_1d) / 3.0
    camarilla_r3 = camarilla_pivot + 1.125 * range_1d
    camarilla_s3 = camarilla_pivot - 1.125 * range_1d
    
    # Calculate 1d Choppiness Index (CHOP)
    # CHOP = 100 * log10(sum(ATR over n) / (n * (max(high) - min(low)))) / log10(n)
    # We'll use a simplified version: CHOP = 100 * log10(atr_sum / (n * range)) / log10(n)
    # Higher CHOP (>61.8) = ranging market, Lower CHOP (<38.2) = trending market
    atr_1d = np.zeros(len(df_1d))
    tr_1d = np.maximum(high_1d[1:] - low_1d[1:], 
                       np.maximum(np.abs(high_1d[1:] - close_1d[:-1]), 
                                  np.abs(low_1d[1:] - close_1d[:-1])))
    # First TR is just high-low
    tr_1d = np.insert(tr_1d, 0, high_1d[0] - low_1d[0])
    
    # Calculate ATR(14) for CHOP
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    
    # Calculate CHOP(14)
    chop_1d = np.full(len(df_1d), np.nan)
    lookback = 14
    for i in range(lookback, len(df_1d)):
        atr_sum = np.sum(atr_1d[i-lookback+1:i+1])
        max_high = np.max(high_1d[i-lookback+1:i+1])
        min_low = np.min(low_1d[i-lookback+1:i+1])
        range_period = max_high - min_low
        if range_period > 0 and atr_sum > 0:
            chop_1d[i] = 100 * np.log10(atr_sum / (lookback * range_period)) / np.log10(lookback)
        else:
            chop_1d[i] = 50.0  # neutral if calculation fails
    
    # Align HTF indicators to 12h timeframe
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Calculate 12h volume confirmation: >2.0x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure sufficient history for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(camarilla_pivot_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(chop_aligned[i]) or
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Chop filter: only trade in ranging markets (CHOP > 61.8)
        chop_filter = chop_aligned[i] > 61.8
        
        # Breakout conditions
        long_breakout = close[i] > camarilla_r3_aligned[i]
        short_breakout = close[i] < camarilla_s3_aligned[i]
        
        # Entry conditions: breakout + volume + chop filter
        long_entry = long_breakout and volume_confirm[i] and chop_filter
        short_entry = short_breakout and volume_confirm[i] and chop_filter
        
        # Exit conditions: price returns to pivot or opposite breakout
        long_exit = close[i] < camarilla_pivot_aligned[i] or short_breakout
        short_exit = close[i] > camarilla_pivot_aligned[i] or long_breakout
        
        # Handle entries and exits
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif (position == 1 and long_exit) or (position == -1 and short_exit):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals