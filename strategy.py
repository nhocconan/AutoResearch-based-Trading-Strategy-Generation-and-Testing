#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian breakout with daily ATR filter and weekly trend filter.
# Donchian breakouts capture momentum in trending markets. ATR filter avoids false breakouts in low volatility.
# Weekly trend filter ensures alignment with higher timeframe direction, improving performance in both bull and bear markets.
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily data for ATR filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate ATR(14) on daily
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = np.zeros(len(close_1d))
    tr1[0] = high_1d[0] - low_1d[0]
    for i in range(1, len(close_1d)):
        tr = max(high_1d[i] - low_1d[i], abs(high_1d[i] - close_1d[i-1]), abs(low_1d[i] - close_1d[i-1]))
        tr1[i] = tr
    atr_1d = np.zeros(len(tr1))
    for i in range(len(tr1)):
        if i < 14:
            atr_1d[i] = np.nan
        else:
            if i == 14:
                atr_1d[i] = np.mean(tr1[0:14])
            else:
                atr_1d[i] = (atr_1d[i-1] * 13 + tr1[i]) / 14
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate 20-period SMA on daily close for trend filter
    sma_20_1d = np.zeros(len(close_1d))
    for i in range(len(close_1d)):
        if i < 20:
            sma_20_1d[i] = np.nan
        else:
            sma_20_1d[i] = np.mean(close_1d[i-20:i])
    sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_20_1d)
    
    # Calculate weekly EMA(20) for trend filter
    close_1w = df_1w['close'].values
    ema_20_1w = np.zeros(len(close_1w))
    ema_multiplier = 2 / (20 + 1)
    if len(close_1w) > 0:
        ema_20_1w[0] = close_1w[0]
        for i in range(1, len(close_1w)):
            ema_20_1w[i] = (close_1w[i] - ema_20_1w[i-1]) * ema_multiplier + ema_20_1w[i-1]
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Calculate 6h Donchian channels (20-period)
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    for i in range(20, n):
        highest_high[i] = np.max(high[i-20:i])
        lowest_low[i] = np.min(low[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):
        # Skip if any required data is not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(atr_1d_aligned[i]) or np.isnan(sma_20_1d_aligned[i]) or 
            np.isnan(ema_20_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        atr = atr_1d_aligned[i]
        sma_20 = sma_20_1d_aligned[i]
        ema_20w = ema_20_1w_aligned[i]
        
        # ATR filter: avoid breakouts in low volatility (ATR < 0.01 * price)
        volatility_filter = atr > 0.01 * price
        
        if position == 0:
            # Long: price breaks above Donchian high with volatility filter and above weekly EMA
            if (price > highest_high[i] and 
                volatility_filter and 
                price > ema_20w):
                position = 1
                signals[i] = position_size
            # Short: price breaks below Donchian low with volatility filter and below weekly EMA
            elif (price < lowest_low[i] and 
                  volatility_filter and 
                  price < ema_20w):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below Donchian low or below daily SMA
            if (price < lowest_low[i] or price < sma_20):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price breaks above Donchian high or above daily SMA
            if (price > highest_high[i] or price > sma_20):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_1d_1w_Donchian_Breakout_ATR_Trend_v1"
timeframe = "6h"
leverage = 1.0