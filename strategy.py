#!/usr/bin/env python3
"""
12h 1D High/Low Breakout with Volume and 1D Trend Filter - Optimized for Lower Trade Frequency
Long: Price breaks above prior 1D high + volume > 2.0x 12h volume MA + price > 1D EMA50 + 12h close > 12h EMA20
Short: Price breaks below prior 1D low + volume > 2.0x 12h volume MA + price < 1D EMA50 + 12h close < 12h EMA20
Exit: Opposite break of prior 1D level
Target: 10-25 trades/year per symbol (50-100 total over 4 years)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1D data for prior high/low and trend filter
    df_1d = get_htf_data(prices, '1d')
    prior_1d_high = df_1d['high'].shift(1)  # Prior day's high
    prior_1d_low = df_1d['low'].shift(1)    # Prior day's low
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    prior_1d_high_aligned = align_htf_to_ltf(prices, df_1d, prior_1d_high.values)
    prior_1d_low_aligned = align_htf_to_ltf(prices, df_1d, prior_1d_low.values)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Get 12h data for volume and trend filters
    df_12h = get_htf_data(prices, '12h')
    volume_ma_20 = pd.Series(df_12h['volume']).rolling(window=20, min_periods=20).mean()
    ema_20_12h = pd.Series(df_12h['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    volume_ma_20_12h = align_htf_to_ltf(prices, df_12h, volume_ma_20.values)
    ema_20_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_20_12h)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    entry_price = 0.0
    
    start_idx = 60  # warmup
    
    for i in range(start_idx, n):
        if (np.isnan(prior_1d_high_aligned[i]) or np.isnan(prior_1d_low_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_ma_20_12h[i]) or
            np.isnan(ema_20_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = volume_ma_20_12h[i]
        
        if position == 0:
            # Long: break above prior 1D high + volume + 1D trend + 12h trend
            if (price > prior_1d_high_aligned[i] and 
                vol > 2.0 * vol_ma and 
                price > ema_50_1d_aligned[i] and 
                close[i] > ema_20_12h_aligned[i]):
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: break below prior 1D low + volume + 1D trend + 12h trend
            elif (price < prior_1d_low_aligned[i] and 
                  vol > 2.0 * vol_ma and 
                  price < ema_50_1d_aligned[i] and 
                  close[i] < ema_20_12h_aligned[i]):
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long exit: break below prior 1D low
            if price < prior_1d_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: break above prior 1D high
            if price > prior_1d_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Prior1D_HL_Breakout_Volume_1DTrend_12hTrend"
timeframe = "12h"
leverage = 1.0