#!/usr/bin/env python3
"""
1h 4h/1d Trend Alignment with Volume Confirmation
Hypothesis: In trending markets, price aligns across timeframes. Use 4h EMA50 for trend direction,
1d EMA200 for higher timeframe filter, and 1h for precise entry timing with volume confirmation.
Volume filter ensures institutional participation. Designed for 60-150 total trades over 4 years.
Works in bull (buy when aligned up) and bear (sell when aligned down) markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4h1d_trend_alignment_v2"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Load 4h data for trend direction (once before loop)
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_prev = np.roll(ema50_4h, 1)
    ema50_4h_prev[0] = ema50_4h[0]
    ema50_rising = ema50_4h > ema50_4h_prev
    ema50_falling = ema50_4h < ema50_4h_prev
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    ema50_rising_aligned = align_htf_to_ltf(prices, df_4h, ema50_rising)
    ema50_falling_aligned = align_htf_to_ltf(prices, df_4h, ema50_falling)
    
    # Load 1d data for higher timeframe filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # 1h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1h EMA21 for entry filter
    ema21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Volume filter: 20-period EMA
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = 200  # For EMA200
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(ema21[i]) or np.isnan(vol_ema[i]) or 
            np.isnan(ema50_4h_aligned[i]) or np.isnan(ema50_rising_aligned[i]) or 
            np.isnan(ema50_falling_aligned[i]) or np.isnan(ema200_1d_aligned[i])):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: trend reversal or stoploss
        if position == 1:  # long position
            # Exit: 4h EMA50 turns down OR price below 1h EMA21 OR stoploss
            if (not ema50_rising_aligned[i] or 
                close[i] < ema21[i] or 
                close[i] <= entry_price - 2.5 * (high[i] - low[i])):  # ATR proxy
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            # Exit: 4h EMA50 turns up OR price above 1h EMA21 OR stoploss
            if (not ema50_falling_aligned[i] or 
                close[i] > ema21[i] or 
                close[i] >= entry_price + 2.5 * (high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
        else:
            # Look for entries: 4h trend + 1d filter + 1h EMA21 + volume
            # Long: 4h uptrend, price above 1d EMA200, price above 1h EMA21, high volume
            long_entry = (ema50_rising_aligned[i] and 
                         close[i] > ema200_1d_aligned[i] and 
                         close[i] > ema21[i] and 
                         volume[i] > vol_ema[i] * 1.5)
            
            # Short: 4h downtrend, price below 1d EMA200, price below 1h EMA21, high volume
            short_entry = (ema50_falling_aligned[i] and 
                          close[i] < ema200_1d_aligned[i] and 
                          close[i] < ema21[i] and 
                          volume[i] > vol_ema[i] * 1.5)
            
            if long_entry:
                signals[i] = 0.20
                position = 1
                entry_price = close[i]
            elif short_entry:
                signals[i] = -0.20
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals