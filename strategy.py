#!/usr/bin/env python3
# 4h_1d_camarilla_pivot_v9
# Hypothesis: 4h price action at 1d Camarilla H3/L3 levels with volume confirmation and 4h EMA trend filter.
# Uses 1d Camarilla H3/L3 as strong support/resistance levels from daily pivot structure.
# Enters long when 4h price bounces above L3 with volume spike and above 4h EMA20.
# Enters short when 4h price reverses below H3 with volume spike and below 4h EMA20.
# Designed for 20-50 trades/year (80-200 over 4 years) with strict entry conditions.
# Works in bull/bear markets: Camarilla levels adapt to volatility, EMA filter ensures trend alignment.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_camarilla_pivot_v9"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for EMA calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:  # Need enough for EMA20
        return np.zeros(n)
    
    # Calculate 4h EMA(20) for trend filter
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=20, adjust=False).mean().values
    
    # Align 4h EMA to 4h timeframe (completed 4h candle only)
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Get 1d HTF data ONCE before loop for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels for daily
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot_1d = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    
    # Camarilla H3/L3 levels (stronger bias filter than H4/L4)
    h3_1d = pivot_1d + (range_1d * 1.1 / 4)
    l3_1d = pivot_1d - (range_1d * 1.1 / 4)
    
    # Align Camarilla levels to 4h timeframe (completed daily candle only)
    h3_1d_aligned = align_htf_to_ltf(prices, df_1d, h3_1d)
    l3_1d_aligned = align_htf_to_ltf(prices, df_1d, l3_1d)
    
    # Volume spike detection (20-period volume average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma_20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_4h_aligned[i]) or np.isnan(h3_1d_aligned[i]) or 
            np.isnan(l3_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below 4h EMA20
            if close[i] < ema_4h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above 4h EMA20
            if close[i] > ema_4h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price closes above 4h EMA20, above 1d L3, with volume spike
            if (close[i] > ema_4h_aligned[i]) and (close[i] > l3_1d_aligned[i]) and vol_spike[i]:
                position = 1
                signals[i] = 0.25
            # Enter short: price closes below 4h EMA20, below 1d H3, with volume spike
            elif (close[i] < ema_4h_aligned[i]) and (close[i] < h3_1d_aligned[i]) and vol_spike[i]:
                position = -1
                signals[i] = -0.25
    
    return signals