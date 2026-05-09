#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_WilliamsFractal_1wTrend_1dPullback"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Williams Fractals on 1d (requires 2-bar confirmation after center bar)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Bearish fractal: high[n-2] < high[n-1] > high[n] and high[n] > high[n+1]
    # Bullish fractal: low[n-2] > low[n-1] < low[n] and low[n] < low[n+1]
    bearish = np.zeros(len(high_1d), dtype=bool)
    bullish = np.zeros(len(low_1d), dtype=bool)
    for i in range(2, len(high_1d)-2):
        if (high_1d[i-2] < high_1d[i-1] and 
            high_1d[i-1] > high_1d[i] and 
            high_1d[i] > high_1d[i+1]):
            bearish[i] = True
        if (low_1d[i-2] > low_1d[i-1] and 
            low_1d[i-1] < low_1d[i] and 
            low_1d[i] < low_1d[i+1]):
            bullish[i] = True
    
    # Align fractals with 2-bar delay for confirmation
    bearish_align = align_htf_to_ltf(prices, df_1d, bearish.astype(float), additional_delay_bars=2)
    bullish_align = align_htf_to_ltf(prices, df_1d, bullish.astype(float), additional_delay_bars=2)
    
    # Weekly trend: EMA50 on 1w
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    ema50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Pullback filter: price near 20-period EMA on 6h
    ema20_6h = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Volume filter: volume > 1.5x 50-period SMA
    vol_ma50 = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    vol_filter = volume > 1.5 * vol_ma50
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if required data unavailable
        if (np.isnan(bearish_align[i]) or np.isnan(bullish_align[i]) or
            np.isnan(ema50_1w_aligned[i]) or np.isnan(ema20_6h[i]) or
            np.isnan(vol_ma50[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: bullish fractal + weekly uptrend + pullback to EMA20 + volume
            if (bullish_align[i] > 0.5 and 
                price > ema50_1w_aligned[i] and 
                abs(price - ema20_6h[i]) / ema20_6h[i] < 0.02 and  # within 2% of EMA20
                vol_filter[i]):
                signals[i] = 0.25
                position = 1
                continue
            
            # Short: bearish fractal + weekly downtrend + pullback to EMA20 + volume
            elif (bearish_align[i] > 0.5 and 
                  price < ema50_1w_aligned[i] and 
                  abs(price - ema20_6h[i]) / ema20_6h[i] < 0.02 and  # within 2% of EMA20
                  vol_filter[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        elif position == 1:
            # Exit long: bearish fractal or breaks below EMA20
            if (bearish_align[i] > 0.5 or 
                price < ema20_6h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: bullish fractal or breaks above EMA20
            if (bullish_align[i] > 0.5 or 
                price > ema20_6h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals