#!/usr/bin/env python3
# 6h_Daily_WMA_Cross_Volume_Regime
# Hypothesis: Daily WMA(9) crossing above/below WMA(21) indicates trend direction on 6h chart.
# Enter long when fast WMA crosses above slow WMA with volume confirmation and price above 6h VWAP.
# Enter short when fast WMA crosses below slow WMA with volume confirmation and price below 6h VWAP.
# Uses daily WMA for trend (updated once per day) and 6h VWAP for entry filtering to avoid whipsaws.
# Works in bull (trend-following with momentum) and bear (mean-reversion during low volatility).

name = "6h_Daily_WMA_Cross_Volume_Regime"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for WMA trend
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily WMA(9) and WMA(21)
    close_1d = df_1d['close'].values
    wma_9 = pd.Series(close_1d).ewm(span=9, adjust=False).mean().values
    wma_21 = pd.Series(close_1d).ewm(span=21, adjust=False).mean().values
    
    # Align daily WMA to 6h chart (wait for daily close)
    wma_9_aligned = align_htf_to_ltf(prices, df_1d, wma_9)
    wma_21_aligned = align_htf_to_ltf(prices, df_1d, wma_21)
    
    # 6h VWAP for entry filtering
    typical_price = (high + low + close) / 3.0
    vwap_num = (typical_price * volume).cumsum()
    vwap_den = volume.cumsum()
    vwap = vwap_num / vwap_den
    
    # Volume confirmation: current volume > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.3 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup
        # Detect WMA crossovers
        wma9_prev = wma_9_aligned[i-1]
        wma21_prev = wma_21_aligned[i-1]
        wma9_curr = wma_9_aligned[i]
        wma21_curr = wma_21_aligned[i]
        
        bullish_cross = wma9_prev <= wma21_prev and wma9_curr > wma21_curr
        bearish_cross = wma9_prev >= wma21_prev and wma9_curr < wma21_curr
        
        if position == 0:
            # LONG: Bullish WMA cross with volume confirmation and price above VWAP
            if bullish_cross and volume_filter[i] and close[i] > vwap[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Bearish WMA cross with volume confirmation and price below VWAP
            elif bearish_cross and volume_filter[i] and close[i] < vwap[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Bearish WMA cross or price breaks below VWAP
            if bearish_cross or close[i] < vwap[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Bullish WMA cross or price breaks above VWAP
            if bullish_cross or close[i] > vwap[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals