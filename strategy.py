#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R1/S1 breakout with volume confirmation and weekly EMA200 trend filter.
# Uses 12h timeframe to reduce trade frequency, weekly trend filter for major trend alignment,
# and volume spike confirmation to avoid false breakouts. Designed for both bull and bear markets.
name = "12h_Camarilla_R1S1_Breakout_1wEMA200_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly data for EMA200 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Daily data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Weekly EMA200 trend filter
    close_1w = df_1w['close'].values
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Previous day's OHLC for Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels: Range = High - Low
    range_1d = high_1d - low_1d
    r1 = close_1d + (range_1d * 1.0833)
    s1 = close_1d - (range_1d * 1.0833)
    
    # Use previous day's levels (shift by 1 to avoid look-ahead)
    r1_shifted = np.roll(r1, 1)
    s1_shifted = np.roll(s1, 1)
    r1_shifted[0] = np.nan
    s1_shifted[0] = np.nan
    
    # Align to 12h timeframe
    r1_12h = align_htf_to_ltf(prices, df_1d, r1_shifted)
    s1_12h = align_htf_to_ltf(prices, df_1d, s1_shifted)
    
    # Volume spike filter: volume > 2.0x 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_spike = volume > (2.0 * vol_ema20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # Wait for weekly EMA200
    
    for i in range(start_idx, n):
        # Skip if required data unavailable
        if (np.isnan(r1_12h[i]) or np.isnan(s1_12h[i]) or
            np.isnan(ema_200_1w_aligned[i]) or np.isnan(vol_ema20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: price breaks above R1 with volume spike and above weekly EMA200
            if (price > r1_12h[i] and vol_spike[i] and price > ema_200_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume spike and below weekly EMA200
            elif (price < s1_12h[i] and vol_spike[i] and price < ema_200_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls back below S1 (mean reversion to support)
            if price < s1_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises back above R1 (mean reversion to resistance)
            if price > r1_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals