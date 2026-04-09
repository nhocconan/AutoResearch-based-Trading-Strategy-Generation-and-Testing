#!/usr/bin/env python3
# 1d_1w_camarilla_breakout_v1
# Hypothesis: Daily breakout of weekly Camarilla levels with weekly EMA20 trend filter and volume confirmation.
# Long when price breaks above weekly H4 resistance with price > weekly EMA20 and volume > 1.5x 20-day average.
# Short when price breaks below weekly L4 support with price < weekly EMA20 and volume > 1.5x 20-day average.
# Exit when price returns to opposite weekly Camarilla level (L4 for longs, H4 for shorts).
# Position size fixed at 0.25 to limit drawdown. Target: 30-100 total trades over 4 years (7-25/year).
# Works in bull markets via breakout continuation and in bear markets via mean reversion at extreme levels.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_camarilla_breakout_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly EMA20 for trend filter
    close_1w = df_1w['close'].values
    ema_20_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 20:
        ema = close_1w[19]  # Initialize with first 20-period average
        multiplier = 2 / (20 + 1)
        ema_20_1w[19] = ema
        for i in range(20, len(close_1w)):
            ema = (close_1w[i] - ema) * multiplier + ema
            ema_20_1w[i] = ema
    
    # Align weekly EMA20 to daily timeframe
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Calculate weekly Camarilla levels from weekly OHLC
    # Camarilla: H4 = C + 1.1*(H-L)/2, L4 = C - 1.1*(H-L)/2
    camarilla_h4_1w = np.full(len(df_1w), np.nan)
    camarilla_l4_1w = np.full(len(df_1w), np.nan)
    for i in range(len(df_1w)):
        c = df_1w['close'].iloc[i]
        h = df_1w['high'].iloc[i]
        l = df_1w['low'].iloc[i]
        camarilla_h4_1w[i] = c + 1.1 * (h - l) / 2
        camarilla_l4_1w[i] = c - 1.1 * (h - l) / 2
    
    # Align weekly Camarilla levels to daily timeframe
    camarilla_h4_1w_aligned = align_htf_to_ltf(prices, df_1w, camarilla_h4_1w)
    camarilla_l4_1w_aligned = align_htf_to_ltf(prices, df_1w, camarilla_l4_1w)
    
    # Volume confirmation: 20-day average
    vol_ma_20 = np.full(n, np.nan)
    vol_sum = 0
    for i in range(n):
        vol_sum += volume[i]
        if i >= 20:
            vol_sum -= volume[i-20]
        if i >= 19:
            vol_ma_20[i] = vol_sum / 20
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(ema_20_1w_aligned[i]) or 
            np.isnan(camarilla_h4_1w_aligned[i]) or 
            np.isnan(camarilla_l4_1w_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price returns to or below L4 level
            if close[i] <= camarilla_l4_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price returns to or above H4 level
            if close[i] >= camarilla_h4_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price breaks above H4 with trend and volume filters
            if (close[i] > camarilla_h4_1w_aligned[i] and 
                close[i] > ema_20_1w_aligned[i] and 
                volume[i] > vol_ma_20[i] * 1.5):
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below L4 with trend and volume filters
            elif (close[i] < camarilla_l4_1w_aligned[i] and 
                  close[i] < ema_20_1w_aligned[i] and 
                  volume[i] > vol_ma_20[i] * 1.5):
                position = -1
                signals[i] = -0.25
    
    return signals