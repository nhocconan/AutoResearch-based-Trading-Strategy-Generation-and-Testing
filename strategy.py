#!/usr/bin/env python3
"""
12h_Williams_Alligator_Signal_With_Volume_Confirmation_v1
Hypothesis: Use Williams Alligator (Jaw:13, Teeth:8, Lips:5 SMAs) to identify trend direction. 
Go long when Lips cross above Teeth and Jaw (bullish alignment) with volume confirmation.
Go short when Lips cross below Teeth and Jaw (bearish alignment) with volume confirmation.
Exit when Alligator lines re-align (Lips between Teeth and Jaw) indicating trend exhaustion.
Designed for low trade frequency (<30/year on 12h) to minimize fee drag while capturing sustained trends in both bull and bear markets.
Williams Alligator excels in trending markets and avoids whipsaws during consolidation.
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
    
    # Williams Alligator components (all SMAs)
    # Jaw: 13-period SMMA (smoothed moving average)
    # Teeth: 8-period SMMA
    # Lips: 5-period SMMA
    def smma(arr, period):
        """Smoothed Moving Average (SMMA)"""
        sma = np.full_like(arr, np.nan, dtype=float)
        if len(arr) >= period:
            sma[period-1] = np.mean(arr[:period])
            for i in range(period, len(arr)):
                sma[i] = (sma[i-1] * (period-1) + arr[i]) / period
        return sma
    
    jaw = smma(close, 13)
    teeth = smma(close, 8)
    lips = smma(close, 5)
    
    # Volume confirmation: >1.5x 20-period average
    vol_ma = np.full_like(volume, np.nan, dtype=float)
    if len(volume) >= 20:
        vol_ma[19] = np.mean(volume[:20])
        for i in range(20, len(volume)):
            vol_ma[i] = np.mean(volume[i-19:i+1])
    volume_spike = volume > (1.5 * vol_ma)
    
    # Weekly trend filter (1w EMA34)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_1w = np.full_like(close_1w, np.nan, dtype=float)
    if len(close_1w) >= 34:
        ema_1w[33] = np.mean(close_1w[:34])
        for i in range(34, len(close_1w)):
            ema_1w[i] = (close_1w[i] * 2 + ema_1w[i-1] * 33) / 35  # EMA approximation
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 30  # Need Alligator components and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(volume_spike[i]) or np.isnan(ema_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Bullish alignment: Lips > Teeth > Jaw
        bullish = lips[i] > teeth[i] and teeth[i] > jaw[i]
        # Bearish alignment: Lips < Teeth < Jaw
        bearish = lips[i] < teeth[i] and teeth[i] < jaw[i]
        
        price = close[i]
        vol_spike = volume_spike[i]
        weekly_trend_up = price > ema_1w_aligned[i]
        weekly_trend_down = price < ema_1w_aligned[i]
        
        if position == 0:
            # Long: bullish alignment + volume spike + above weekly EMA
            if bullish and vol_spike and weekly_trend_up:
                signals[i] = 0.25
                position = 1
            # Short: bearish alignment + volume spike + below weekly EMA
            elif bearish and vol_spike and weekly_trend_down:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: Alligator re-aligns (Lips between Teeth and Jaw) or below weekly EMA
            if not bullish or price < ema_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: Alligator re-aligns (Lips between Teeth and Jaw) or above weekly EMA
            if not bearish or price > ema_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Williams_Alligator_Signal_With_Volume_Confirmation_v1"
timeframe = "12h"
leverage = 1.0