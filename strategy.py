#!/usr/bin/env python3
"""
4h Donchian Breakout with Volume Confirmation and Trend Filter (HMA)
Hypothesis: Donchian channel breakouts on 4h timeframe capture genuine momentum moves.
Volume confirmation ensures institutional participation, while 1d HMA trend filter ensures
we only trade in the direction of higher timeframe trend, reducing false breakouts.
This combination works in both bull and bear markets by focusing on breakouts with
volume and trend alignment. Target: 20-40 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian Channel (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    upper = high_series.rolling(window=20, min_periods=20).max()
    lower = low_series.rolling(window=20, min_periods=20).min()
    
    # Volume filter: 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    # 1d HMA trend filter (Hull Moving Average)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 21:
        hma_1d = np.full(len(prices), np.nan)
    else:
        # Calculate HMA: WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
        def wma(values, window):
            weights = np.arange(1, window + 1)
            return np.convolve(values, weights/weights.sum(), mode='valid')
        
        close_1d = df_1d['close'].values
        n = 21
        half_n = n // 2
        sqrt_n = int(np.sqrt(n))
        
        wma_full = wma(close_1d, n)
        wma_half = wma(close_1d, half_n)
        wma_2x_minus = 2 * wma_half - wma_full
        hma_raw = wma(wma_2x_minus, sqrt_n)
        
        # Align to lower timeframe with proper delay
        hma_1d_raw = np.full(len(close_1d), np.nan)
        hma_1d_raw[sqrt_n-1:sqrt_n-1+len(hma_raw)] = hma_raw
        hma_1d = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Warmup for Donchian and volume
    
    for i in range(start_idx, n):
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(volume_filter[i]) or np.isnan(hma_1d[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        upper_band = upper[i]
        lower_band = lower[i]
        vol_ok = volume_filter[i]
        hma_trend = hma_1d[i]
        
        if position == 0:
            # Long: break above upper Donchian with volume and uptrend
            if price > upper_band and vol_ok and price > hma_trend:
                signals[i] = 0.25
                position = 1
            # Short: break below lower Donchian with volume and downtrend
            elif price < lower_band and vol_ok and price < hma_trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long if price returns to lower band or trend changes
            if price < lower_band or price < hma_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short if price returns to upper band or trend changes
            if price > upper_band or price > hma_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian_Breakout_Volume_HMA_Trend"
timeframe = "4h"
leverage = 1.0