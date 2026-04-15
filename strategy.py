#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1-day ATR Filter
# Uses Williams Alligator (Jaw/Teeth/Lips) on 12h timeframe to identify trends.
# Enters long when Lips > Teeth > Jaw (bullish alignment) and price > Lips,
# enters short when Lips < Teeth < Jaw (bearish alignment) and price < Lips.
# Filters trades using 1-day ATR volatility: only trade when ATR(14) > median ATR(50) to avoid low-volatility chop.
# Works in bull markets (captures uptrends) and bear markets (captures downtrends).
# Target: 50-150 total trades over 4 years (12-37/year).

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data for Williams Alligator
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Load 1d data for ATR filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams Alligator on 12h: SMMA (Smoothed Moving Average)
    # Jaw: 13-period SMMA of median price, shifted 8 bars
    # Teeth: 8-period SMMA of median price, shifted 5 bars
    # Lips: 5-period SMMA of median price, shifted 3 bars
    median_price_12h = (high_12h + low_12h) / 2
    
    def smma(arr, period):
        """Smoothed Moving Average"""
        result = np.full_like(arr, np.nan)
        if len(arr) < period:
            return result
        # First value is simple SMA
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CURRENT) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(median_price_12h, 13)
    teeth = smma(median_price_12h, 8)
    lips = smma(median_price_12h, 5)
    
    # Shift the lines as per Alligator definition
    jaw = np.roll(jaw, 8)
    teeth = np.roll(teeth, 5)
    lips = np.roll(lips, 3)
    
    # Align Alligator lines to 12h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_12h, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips)
    
    # ATR on 1d for volatility filter
    def calculate_atr(high, low, close, period):
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]  # First value
        atr = np.full_like(tr, np.nan)
        for i in range(period, len(tr)):
            if i == period:
                atr[i] = np.mean(tr[:period])
            else:
                atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        return atr
    
    atr_1d = calculate_atr(high_1d, low_1d, close_1d, 14)
    atr_ma_1d = np.full_like(atr_1d, np.nan)
    for i in range(50, len(atr_1d)):
        if i == 50:
            atr_ma_1d[i] = np.mean(atr_1d[:50])
        else:
            atr_ma_1d[i] = (atr_ma_1d[i-1] * 49 + atr_1d[i]) / 50
    
    # Align ATR and its MA to 12h timeframe
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    atr_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_ma_1d)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(atr_1d_aligned[i]) or 
            np.isnan(atr_ma_1d_aligned[i])):
            continue
        
        # Volatility filter: only trade when ATR > MA(ATR) (avoid low-volatility chop)
        vol_filter = atr_1d_aligned[i] > atr_ma_1d_aligned[i]
        
        # Bullish alignment: Lips > Teeth > Jaw
        bullish_alignment = lips_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > jaw_aligned[i]
        # Bearish alignment: Lips < Teeth < Jaw
        bearish_alignment = lips_aligned[i] < teeth_aligned[i] and teeth_aligned[i] < jaw_aligned[i]
        
        # Long entry: bullish alignment + price above Lips + volatility filter
        if (bullish_alignment and close[i] > lips_aligned[i] and vol_filter and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: bearish alignment + price below Lips + volatility filter
        elif (bearish_alignment and close[i] < lips_aligned[i] and vol_filter and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: opposite alignment or volatility filter fails
        elif position == 1 and (not bullish_alignment or not vol_filter):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (not bearish_alignment or not vol_filter):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "12h_Williams_Alligator_ATR_Filter"
timeframe = "12h"
leverage = 1.0