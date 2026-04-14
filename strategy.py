#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Camarilla pivot reversal with 1-day trend filter and volume confirmation
# Long when price touches Camarilla L3 level with volume >1.5x average and price above 1d EMA200
# Short when price touches Camarilla H3 level with volume >1.5x average and price below 1d EMA200
# Exit when price crosses Camarilla H4/L4 levels (strong reversal)
# 1-day EMA200 acts as trend filter to avoid counter-trend trades
# Target: 75-200 total trades over 4 years (19-50/year) to balance opportunity and fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 4h and 1d data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h Camarilla pivot levels from 4h OHLC
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate pivot point and ranges
    pivot = (high_4h + low_4h + close_4h) / 3
    range_4h = high_4h - low_4h
    
    # Camarilla levels
    h4 = close_4h + (range_4h * 1.1 / 2)
    l4 = close_4h - (range_4h * 1.1 / 2)
    h3 = close_4h + (range_4h * 1.1 / 4)
    l3 = close_4h - (range_4h * 1.1 / 4)
    h2 = close_4h + (range_4h * 1.1 / 6)
    l2 = close_4h - (range_4h * 1.1 / 6)
    h1 = close_4h + (range_4h * 1.1 / 12)
    l1 = close_4h - (range_4h * 1.1 / 12)
    
    # Calculate 1d EMA200
    close_1d = df_1d['close'].values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, min_periods=200, adjust=False).mean().values
    
    # Calculate 4h volume average (20-period)
    vol_4h = df_4h['volume'].values
    vol_ma_4h = pd.Series(vol_4h).rolling(window=20, min_periods=20).mean().values
    
    # Align indicators to 4h timeframe
    h4_aligned = align_htf_to_ltf(prices, df_4h, h4)
    l4_aligned = align_htf_to_ltf(prices, df_4h, l4)
    h3_aligned = align_htf_to_ltf(prices, df_4h, h3)
    l3_aligned = align_htf_to_ltf(prices, df_4h, l3)
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    vol_ma_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_4h)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 50  # for 20-period calculations and EMA200
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or 
            np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or
            np.isnan(ema_200_1d_aligned[i]) or np.isnan(vol_ma_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_4h_current = volume[i]  # Current 4h volume
        
        if position == 0:
            # Long setup: touch L3 with volume confirmation and price above 1d EMA200
            if (price <= l3_aligned[i] and 
                vol_4h_current > 1.5 * vol_ma_4h_aligned[i] and  # Volume confirmation
                price > ema_200_1d_aligned[i]):                 # Price above 1d EMA200 for bullish bias
                position = 1
                signals[i] = position_size
            # Short setup: touch H3 with volume confirmation and price below 1d EMA200
            elif (price >= h3_aligned[i] and 
                  vol_4h_current > 1.5 * vol_ma_4h_aligned[i] and  # Volume confirmation
                  price < ema_200_1d_aligned[i]):                 # Price below 1d EMA200 for bearish bias
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses above H4 (strong reversal)
            if price >= h4_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses below L4 (strong reversal)
            if price <= l4_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_Camarilla_L3H3_1dEMA200_Volume"
timeframe = "4h"
leverage = 1.0