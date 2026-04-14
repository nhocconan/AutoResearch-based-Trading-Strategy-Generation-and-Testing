# 12h_1d_KAMA_Trend_Filter
# Hypothesis: Kauffman Adaptive Moving Average (KAMA) on 1d timeframe provides a robust trend filter.
# Trades occur at 12h timeframe when price crosses KAMA in the direction of the trend, with volume confirmation.
# This adaptive moving average reduces whipsaw during sideways markets while capturing strong trends in both bull and bear markets.
# Target: 20-40 trades/year to minimize fee drag.

#!/usr/bin/env python3
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
    
    # Load daily data (HTF) once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1-day KAMA (10-period efficiency ratio, 2/30 fast/slow)
    kama_1d = np.full(len(df_1d), np.nan)
    if len(df_1d) >= 30:
        # Initialize
        kama_1d[0] = close_1d[0]
        # Precompute smoothing constants
        fast_sc = 2 / (2 + 1)   # EMA(2)
        slow_sc = 2 / (30 + 1)  # EMA(30)
        
        for i in range(1, len(df_1d)):
            # Direction: absolute price change over lookback period
            direction = abs(close_1d[i] - close_1d[i-10]) if i >= 10 else abs(close_1d[i] - close_1d[0])
            # Volatility: sum of absolute price changes over lookback period
            volatility = 0
            for j in range(1, 11):
                if i - j >= 0:
                    volatility += abs(close_1d[i-j+1] - close_1d[i-j])
            if volatility == 0:
                er = 0
            else:
                er = direction / volatility
            # Smoothing constant
            sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
            # KAMA calculation
            kama_1d[i] = kama_1d[i-1] + sc * (close_1d[i] - kama_1d[i-1])
    
    kama_12h = align_htf_to_ltf(prices, df_1d, kama_1d)
    
    # Calculate daily volume moving average (20-period)
    vol_ma_20_1d = np.full(len(df_1d), np.nan)
    if len(df_1d) >= 20:
        for i in range(19, len(df_1d)):
            vol_ma_20_1d[i] = np.mean(volume_1d[i-19:i+1])
    
    vol_ma_20_12h = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Calculate 12h price change for momentum filter
    price_change_12h = np.full(n, np.nan)
    if n >= 2:
        price_change_12h[1:] = (close[1:] - close[:-1]) / close[:-1]
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    for i in range(30, n):
        # Skip if any critical data is NaN
        if (np.isnan(kama_12h[i]) or
            np.isnan(vol_ma_20_12h[i]) or
            np.isnan(price_change_12h[i])):
            signals[i] = 0.0
            continue
        
        # Volume ratio: current volume vs 20-period average
        if vol_ma_20_12h[i] <= 0:
            volume_ratio = 0
        else:
            volume_ratio = volume[i] / vol_ma_20_12h[i]
        
        # Volume threshold: require significant spike
        vol_threshold = 1.8
        
        if position == 0:
            # Long: Price crosses above KAMA with volume confirmation and positive momentum
            if close[i] > kama_12h[i] and volume_ratio > vol_threshold and price_change_12h[i] > 0:
                position = 1
                signals[i] = position_size
            # Short: Price crosses below KAMA with volume confirmation and negative momentum
            elif close[i] < kama_12h[i] and volume_ratio > vol_threshold and price_change_12h[i] < 0:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit: Price crosses back below KAMA
            if close[i] < kama_12h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit: Price crosses back above KAMA
            if close[i] > kama_12h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1d_KAMA_Trend_Filter"
timeframe = "12h"
leverage = 1.0