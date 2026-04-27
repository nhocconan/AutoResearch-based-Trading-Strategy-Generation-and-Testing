#!/usr/bin/env python3
"""
1d_WickReversal_MeanReversion_With_Volume
Mean reversion strategy on 1d timeframe: Long when price closes below open (bearish candle) with volume > 1.5x average, short when price closes above open (bullish candle) with volume > 1.5x average.
Exit on opposite signal or when price returns to open level.
Uses 1w EMA200 as trend filter: only take longs above 1w EMA200, shorts below 1w EMA200.
Designed to work in both bull and bear markets by fading extreme moves with volume confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    open_price = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA200 for trend filter
    ema_period = 200
    ema_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= ema_period:
        ema_1w[ema_period - 1] = np.mean(close_1w[:ema_period])
        for i in range(ema_period, len(close_1w)):
            ema_1w[i] = (close_1w[i] * (2 / (ema_period + 1)) + 
                         ema_1w[i - 1] * (1 - (2 / (ema_period + 1))))
    
    # Align 1w EMA200 to daily timeframe
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Volume MA for confirmation (20-period)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i - 19:i + 1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need EMA1w, volume MA20
    start_idx = max(ema_period - 1, 19)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_1w_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        price_open = open_price[i]
        vol_now = volume[i]
        vol_avg = vol_ma_20[i]
        
        # Volume filter: require volume above average
        vol_filter = vol_now > 1.5 * vol_avg
        
        # Wick reversal signals: close vs open
        is_bearish = price_close < price_open  # bearish candle
        is_bullish = price_close > price_open   # bullish candle
        
        if position == 0:
            # Long: bearish candle with volume filter and price above 1w EMA200
            if is_bearish and vol_filter and price_close > ema_1w_aligned[i]:
                signals[i] = size
                position = 1
            # Short: bullish candle with volume filter and price below 1w EMA200
            elif is_bullish and vol_filter and price_close < ema_1w_aligned[i]:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: bullish candle or price returns to/open above open
            if is_bullish or price_close >= price_open:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: bearish candle or price returns to/below open
            if is_bearish or price_close <= price_open:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_WickReversal_MeanReversion_With_Volume"
timeframe = "1d"
leverage = 1.0