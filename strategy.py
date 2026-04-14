#!/usr/bin/env python3
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
    
    # Load daily data for ATR and close
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate daily ATR(14)
    def calculate_atr(high, low, close, period):
        tr1 = np.abs(high - low)
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]  # First TR is just high-low
        atr = np.full_like(close, np.nan)
        if len(close) >= period:
            atr[period-1] = np.mean(tr[:period])
            for i in range(period, len(close)):
                atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        return atr
    
    atr_14_1d = calculate_atr(high_1d, low_1d, close_1d, 14)
    
    # Calculate daily SMA(200) for trend filter
    sma_200_1d = np.full_like(close_1d, np.nan)
    if len(close_1d) >= 200:
        for i in range(199, len(close_1d)):
            sma_200_1d[i] = np.mean(close_1d[i-199:i+1])
    
    # Align 1d indicators to 12h timeframe
    atr_14_1d_12h = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    sma_200_1d_12h = align_htf_to_ltf(prices, df_1d, sma_200_1d)
    
    # Calculate ATR-based channel on 12h
    atr_mult = 2.0
    upper_channel = np.full(n, np.nan)
    lower_channel = np.full(n, np.nan)
    
    for i in range(1, n):
        upper_channel[i] = close[i-1] + atr_14_1d_12h[i-1] * atr_mult
        lower_channel[i] = close[i-1] - atr_14_1d_12h[i-1] * atr_mult
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    for i in range(200, n):
        # Skip if any critical data is NaN
        if (np.isnan(atr_14_1d_12h[i-1]) or 
            np.isnan(sma_200_1d_12h[i]) or
            np.isnan(upper_channel[i]) or
            np.isnan(lower_channel[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Price breaks above upper channel AND above daily SMA200
            if close[i] > upper_channel[i] and close[i] > sma_200_1d_12h[i]:
                position = 1
                signals[i] = position_size
            # Short: Price breaks below lower channel AND below daily SMA200
            elif close[i] < lower_channel[i] and close[i] < sma_200_1d_12h[i]:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit: Price crosses below lower channel OR below daily SMA200
            if close[i] < lower_channel[i] or close[i] < sma_200_1d_12h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit: Price crosses above upper channel OR above daily SMA200
            if close[i] > upper_channel[i] or close[i] > sma_200_1d_12h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1d_ATR_Channel_SMA200"
timeframe = "12h"
leverage = 1.0