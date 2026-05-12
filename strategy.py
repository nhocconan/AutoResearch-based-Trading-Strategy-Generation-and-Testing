#!/usr/bin/env python3
"""
12h_MonetaryPulse_With_Volume_Regime
Hypothesis: 12-hour strategy combining monetary pulse (price change over 3 periods) with volume confirmation and Choppiness Index regime filter.
Only takes long when monetary pulse is positive, volume > 2x average, and market is trending (CHOP < 38.2).
Short when monetary pulse negative, volume spike, and trending.
Uses 1d EMA200 as additional trend filter to avoid counter-trend trades in strong trends.
Designed for low trade frequency (target: 15-25/year) to minimize fee drag while capturing sustained moves.
Works in bull/bear via regime filter and dual trend confirmation.
"""

name = "12h_MonetaryPulse_With_Volume_Regime"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Monetary pulse: price change over 3 periods (normalized)
    price_change = (close - np.roll(close, 3)) / np.roll(close, 3)
    price_change[0:3] = 0  # first 3 values invalid
    
    # Volume confirmation: >2.0x 50-period average
    vol_ma = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # Choppiness Index regime filter (14-period)
    def calculate_chop(high, low, close, window=14):
        atr = []
        for i in range(len(high)):
            if i == 0:
                tr = high[i] - low[i]
            else:
                tr = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
            atr.append(tr)
        
        atr = np.array(atr)
        tr_sum = pd.Series(atr).rolling(window=window, min_periods=window).sum().values
        hh = pd.Series(high).rolling(window=window, min_periods=window).max().values
        ll = pd.Series(low).rolling(window=window, min_periods=window).min().values
        denominator = hh - ll
        chop = np.where(denominator != 0, 100 * np.log10(tr_sum / denominator) / np.log10(window), 50)
        return chop
    
    chop = calculate_chop(high, low, close, 14)
    trending = chop < 38.2  # trending regime
    
    # 1d EMA200 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    ema_200_1d = pd.Series(df_1d['close']).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        if (np.isnan(ema_200_1d_aligned[i]) or 
            np.isnan(chop[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Positive monetary pulse + volume spike + trending + price above 1d EMA200
            if (price_change[i] > 0.01 and 
                volume_spike[i] and 
                trending[i] and 
                close[i] > ema_200_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Negative monetary pulse + volume spike + trending + price below 1d EMA200
            elif (price_change[i] < -0.01 and 
                  volume_spike[i] and 
                  trending[i] and 
                  close[i] < ema_200_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Monetary pulse turns negative OR price below 1d EMA200 OR chop > 61.8 (ranging)
            if (price_change[i] < 0 or 
                close[i] < ema_200_1d_aligned[i] or 
                chop[i] > 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Monetary pulse turns positive OR price above 1d EMA200 OR chop > 61.8 (ranging)
            if (price_change[i] > 0 or 
                close[i] > ema_200_1d_aligned[i] or 
                chop[i] > 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals