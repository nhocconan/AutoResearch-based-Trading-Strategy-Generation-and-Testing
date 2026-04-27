#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    """
    1d Williams Alligator + 1week trend filter with volume confirmation.
    Long when: Jaw > Teeth > Lips (bullish alignment) + price > weekly EMA20 + volume > 1.5x average
    Short when: Jaw < Teeth < Lips (bearish alignment) + price < weekly EMA20 + volume > 1.5x average
    Exit when: Alligator lines cross or price crosses weekly EMA20
    Designed for low-frequency swing trading on daily timeframe to avoid overtrading.
    """
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly EMA20 for trend filter
    close_1w = df_1w['close'].values
    ema_20_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 20:
        ema_20_1w[19] = np.mean(close_1w[:20])
        for i in range(20, len(close_1w)):
            ema_20_1w[i] = (close_1w[i] * 2 + ema_20_1w[i-1] * 18) / 20  # EMA20
    
    # Get daily data for Williams Alligator
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    # Calculate Williams Alligator lines (Jaw, Teeth, Lips) on daily data
    # Jaw: 13-period SMMA shifted 8 bars
    # Teeth: 8-period SMMA shifted 5 bars  
    # Lips: 5-period SMMA shifted 3 bars
    median_price_1d = (df_1d['high'].values + df_1d['low'].values) / 2
    
    def smma(arr, period):
        """Smoothed Moving Average"""
        result = np.full(len(arr), np.nan)
        if len(arr) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (PERIOD-1) + CURRENT) / PERIOD
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(median_price_1d, 13)
    teeth = smma(median_price_1d, 8)
    lips = smma(median_price_1d, 5)
    
    # Shift the lines as per Alligator definition
    jaw = np.roll(jaw, 8)
    teeth = np.roll(teeth, 5)
    lips = np.roll(lips, 3)
    
    # Align weekly and daily indicators to 1d timeframe
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Calculate daily ATR(14) for volatility filter
    tr = np.maximum(high[1:] - low[1:], 
                    np.maximum(np.abs(high[1:] - close[:-1]), 
                               np.abs(low[1:] - close[:-1])))
    tr = np.concatenate([[np.nan], tr])
    atr = np.full(n, np.nan)
    for i in range(14, n):
        if i == 14:
            atr[i] = np.mean(tr[1:15])
        else:
            atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # Calculate 20-period volume average
    vol_ma = np.full(n, np.nan)
    vol_period = 20
    for i in range(vol_period, n):
        vol_ma[i] = np.mean(volume[i-vol_period:i])
    
    signals = np.zeros(n)
    position = 0
    size = 0.25
    
    # Warmup period
    start_idx = max(20, 13, 14, vol_period) + 10
    
    for i in range(start_idx, n):
        if (np.isnan(ema_20_1w_aligned[i]) or np.isnan(jaw_aligned[i]) or 
            np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or 
            np.isnan(atr[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        # Volume filter: at least 1.5x average volume
        vol_filter = vol_ratio > 1.5
        
        # Alligator alignment
        bullish_alignment = jaw_aligned[i] > teeth_aligned[i] > lips_aligned[i]
        bearish_alignment = jaw_aligned[i] < teeth_aligned[i] < lips_aligned[i]
        
        if position == 0:
            # Long: Bullish Alligator alignment + price above weekly EMA20 + volume
            if bullish_alignment and price > ema_20_1w_aligned[i] and vol_filter:
                signals[i] = size
                position = 1
            # Short: Bearish Alligator alignment + price below weekly EMA20 + volume
            elif bearish_alignment and price < ema_20_1w_aligned[i] and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: Alligator lines cross or price crosses below weekly EMA20
            if not bullish_alignment or price < ema_20_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: Alligator lines cross or price crosses above weekly EMA20
            if not bearish_alignment or price > ema_20_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_WilliamsAlligator_1wEMA20_VolumeFilter"
timeframe = "1d"
leverage = 1.0