#!/usr/bin/env python3
"""
12h Williams Alligator with Volume Confirmation and 1D Trend Filter
Long: Jaw < Teeth < Lips (bullish alignment) + volume > 1.5x 12h volume MA + price > 1D EMA50
Short: Jaw > Teeth > Lips (bearish alignment) + volume > 1.5x 12h volume MA + price < 1D EMA50
Exit: Opposite Alligator alignment or price crosses Jaw
Williams Alligator (13,8,5 SMAs) provides trend direction with built-in smoothing
Targets 15-25 trades/year per symbol with strong trend filtering to avoid whipsaw
"""

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
    
    # Get 1D data for trend filter
    df_1d = get_htf_data(prices, '1d')
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Williams Alligator on 12h: Jaw (13), Teeth (8), Lips (5) SMAs
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean()
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean()
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean()
    
    # 12h volume moving average for confirmation
    df_12h = get_htf_data(prices, '12h')
    volume_ma_4 = pd.Series(df_12h['volume']).rolling(window=4, min_periods=4).mean()
    volume_ma_4_12h = align_htf_to_ltf(prices, df_12h, volume_ma_4.values)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 50  # warmup for all indicators
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_ma_4_12h[i]) or
            np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = volume_ma_4_12h[i]
        
        # Williams Alligator alignment
        bullish_alignment = jaw[i] < teeth[i] < lips[i]
        bearish_alignment = jaw[i] > teeth[i] > lips[i]
        
        if position == 0:
            # Long: bullish alignment + volume + 1D trend
            if bullish_alignment and vol > 1.5 * vol_ma and price > ema_50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: bearish alignment + volume + 1D trend
            elif bearish_alignment and vol > 1.5 * vol_ma and price < ema_50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: bearish alignment or price crosses below Jaw
            if bearish_alignment or price < jaw[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: bullish alignment or price crosses above Jaw
            if bullish_alignment or price > jaw[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsAlligator_Volume_1DTrend"
timeframe = "12h"
leverage = 1.0