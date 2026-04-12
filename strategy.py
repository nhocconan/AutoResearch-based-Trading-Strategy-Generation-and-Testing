#!/usr/bin/env python3
"""
4h_1d_camarilla_breakout_volume
Camarilla pivot levels on 1d timeframe for support/resistance levels.
Breakout from levels with volume confirmation and chop regime filter.
Designed for low trade frequency (target: 20-35 trades/year) to minimize fee drag.
Works in both trending and ranging markets: breakouts in trends, mean reversion at extremes in ranges.
"""

name = "4h_1d_camarilla_breakout_volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous day
    high_prev = df_1d['high'].shift(1).values
    low_prev = df_1d['low'].shift(1).values
    close_prev = df_1d['close'].shift(1).values
    
    # Camarilla levels
    # H5 = Close + 1.1*(High-Low)*1.1/2
    # H4 = Close + 1.1*(High-Low)*1.1/4
    # H3 = Close + 1.1*(High-Low)*1.1/6
    # L3 = Close - 1.1*(High-Low)*1.1/6
    # L4 = Close - 1.1*(High-Low)*1.1/4
    # L5 = Close - 1.1*(High-Low)*1.1/2
    
    camarilla_h4 = close_prev + 1.1 * (high_prev - low_prev) * 1.1 / 4
    camarilla_l4 = close_prev - 1.1 * (high_prev - low_prev) * 1.1 / 4
    camarilla_h3 = close_prev + 1.1 * (high_prev - low_prev) * 1.1 / 6
    camarilla_l3 = close_prev - 1.1 * (high_prev - low_prev) * 1.1 / 6
    camarilla_h5 = close_prev + 1.1 * (high_prev - low_prev) * 1.1 / 2
    camarilla_l5 = close_prev - 1.1 * (high_prev - low_prev) * 1.1 / 2
    
    # Align Camarilla levels to 4h timeframe
    h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    h5_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h5)
    l5_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l5)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    # Chop regime filter: Chop > 61.8 = ranging (mean revert), Chop < 38.2 = trending (breakout)
    atr_period = 14
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    # Calculate Chop: 100 * log10(sum(TR)/ (ATR * n)) / log10(n)
    chop_period = 14
    sum_tr = pd.Series(tr).rolling(window=chop_period, min_periods=chop_period).sum().values
    chop = 100 * np.log10(sum_tr / (atr * chop_period)) / np.log10(chop_period)
    chop[~np.isfinite(chop)] = 50  # handle invalid values
    
    chop_threshold_high = 61.8  # ranging
    chop_threshold_low = 38.2   # trending
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if data not ready
        if (np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or 
            np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or
            np.isnan(h5_aligned[i]) or np.isnan(l5_aligned[i]) or
            np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        # Long entry: break above H4 in trending market OR bounce from L3 in ranging market
        long_breakout = (close[i] > h4_aligned[i] and chop[i] < chop_threshold_low and vol_confirm[i])
        long_bounce = (close[i] < l3_aligned[i] and chop[i] > chop_threshold_high and 
                       close[i] > l4_aligned[i] and vol_confirm[i])
        
        # Short entry: break below L4 in trending market OR bounce from H3 in ranging market
        short_breakout = (close[i] < l4_aligned[i] and chop[i] < chop_threshold_low and vol_confirm[i])
        short_bounce = (close[i] > h3_aligned[i] and chop[i] > chop_threshold_high and 
                        close[i] < h4_aligned[i] and vol_confirm[i])
        
        # Entry logic
        if (long_breakout or long_bounce) and position != 1:
            position = 1
            signals[i] = 0.25
        elif (short_breakout or short_bounce) and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit conditions
        elif position == 1 and (close[i] < l3_aligned[i] or close[i] > h5_aligned[i]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] > h3_aligned[i] or close[i] < l5_aligned[i]):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals