#!/usr/bin/env python3
"""
12h Volatility-Adjusted Breakout with Volume and Trend Filter
Long: Price breaks above 1D high + volume > 1.8x 12h vol MA + 1D EMA50 > 1D EMA200
Short: Price breaks below 1D low + volume > 1.8x 12h vol MA + 1D EMA50 < 1D EMA200
Exit: Opposite break of 1D level
Targets 15-25 trades/year by tightening volume threshold and adding EMA filter
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1D data for levels and trend filters
    df_1d = get_htf_data(prices, '1d')
    prior_1d_high = df_1d['high'].shift(1)
    prior_1d_low = df_1d['low'].shift(1)
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_200_1d = pd.Series(df_1d['close']).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    prior_1d_high_aligned = align_htf_to_ltf(prices, df_1d, prior_1d_high.values)
    prior_1d_low_aligned = align_htf_to_ltf(prices, df_1d, prior_1d_low.values)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # 12h volume moving average (20-period for stability)
    df_12h = get_htf_data(prices, '12h')
    volume_ma_20 = pd.Series(df_12h['volume']).rolling(window=20, min_periods=20).mean()
    volume_ma_20_12h = align_htf_to_ltf(prices, df_12h, volume_ma_20.values)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 200
    
    for i in range(start_idx, n):
        if (np.isnan(prior_1d_high_aligned[i]) or np.isnan(prior_1d_low_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(ema_200_1d_aligned[i]) or 
            np.isnan(volume_ma_20_12h[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = volume_ma_20_12h[i]
        
        if position == 0:
            # Long: break above 1D high + volume spike + EMA50 > EMA200 (bullish bias)
            if price > prior_1d_high_aligned[i] and vol > 1.8 * vol_ma and ema_50_1d_aligned[i] > ema_200_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below 1D low + volume spike + EMA50 < EMA200 (bearish bias)
            elif price < prior_1d_low_aligned[i] and vol > 1.8 * vol_ma and ema_50_1d_aligned[i] < ema_200_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: break below 1D low
            if price < prior_1d_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: break above 1D high
            if price > prior_1d_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_VolAdjusted_Breakout_Volume_EMAFilter"
timeframe = "12h"
leverage = 1.0