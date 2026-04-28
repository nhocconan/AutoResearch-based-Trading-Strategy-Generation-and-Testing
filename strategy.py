#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using 1w Camarilla R3/S3 breakout with volume confirmation and chop regime filter.
# Enter long when price breaks above 1w Camarilla R3 with volume > 2.0x 50-bar average and chop > 61.8 (range).
# Enter short when price breaks below 1w Camarilla S3 with volume > 2.0x 50-bar average and chop > 61.8 (range).
# Exit on opposite 1w Camarilla level (R2/S2).
# Uses discrete position sizing (0.25) to control risk. Target: 30-100 total trades over 4 years.
# Camarilla provides mathematically derived support/resistance, volume confirms breakout strength,
# chop filter ensures mean-reversion logic works in ranging markets (common in 2025+ bear/range regime).

name = "1d_Camarilla_R3S3_Breakout_1wVolumeChop_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for Camarilla calculation (HTF)
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate 1w Camarilla levels (using previous week's OHLC)
    h_1w = df_1w['high'].values
    l_1w = df_1w['low'].values
    c_1w = df_1w['close'].values
    
    typical_price = (h_1w + l_1w + c_1w) / 3.0
    hl_range = h_1w - l_1w
    
    r3_1w = typical_price + (hl_range * 1.1 / 4.0)
    s3_1w = typical_price - (hl_range * 1.1 / 4.0)
    r2_1w = typical_price + (hl_range * 1.1 / 6.0)
    s2_1w = typical_price - (hl_range * 1.1 / 6.0)
    
    # Align 1w Camarilla levels to 1d timeframe
    r3_1w_aligned = align_htf_to_ltf(prices, df_1w, r3_1w)
    s3_1w_aligned = align_htf_to_ltf(prices, df_1w, s3_1w)
    r2_1w_aligned = align_htf_to_ltf(prices, df_1w, r2_1w)
    s2_1w_aligned = align_htf_to_ltf(prices, df_1w, s2_1w)
    
    # Calculate 1d chop regime filter (choppiness index > 61.8 = ranging)
    def calculate_chop(high, low, close, window=14):
        atr = np.zeros(len(close))
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]  # First period TR is just high-low
        atr = pd.Series(tr).rolling(window=window, min_periods=window).mean().values
        high_max = pd.Series(high).rolling(window=window, min_periods=window).max().values
        low_min = pd.Series(low).rolling(window=window, min_periods=window).min().values
        chop = np.where((high_max - low_min) > 0, 
                        100 * np.log10(np.sum(atr, axis=0) / np.log(window) / (high_max - low_min)), 
                        50)
        return chop
    
    chop = calculate_chop(high, low, close, 14)
    chop_filter = chop > 61.8  # Range regime
    
    # Calculate 1d volume confirmation: >2.0x 50-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_50 = volume_series.rolling(window=50, min_periods=50).mean().values
    volume_confirm = volume > 2.0 * volume_ma_50
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure sufficient history for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r3_1w_aligned[i]) or np.isnan(s3_1w_aligned[i]) or 
            np.isnan(r2_1w_aligned[i]) or np.isnan(s2_1w_aligned[i]) or
            np.isnan(chop[i]) or np.isnan(volume_ma_50[i])):
            signals[i] = 0.0
            continue
        
        # Camarilla breakout conditions with volume confirmation and chop filter
        long_breakout = close[i] > r3_1w_aligned[i] and volume_confirm[i] and chop_filter[i]
        short_breakout = close[i] < s3_1w_aligned[i] and volume_confirm[i] and chop_filter[i]
        
        # Exit conditions: opposite Camarilla level (R2/S2)
        long_exit = close[i] < r2_1w_aligned[i]
        short_exit = close[i] > s2_1w_aligned[i]
        
        # Handle entries and exits
        if long_breakout and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_breakout and position >= 0:
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