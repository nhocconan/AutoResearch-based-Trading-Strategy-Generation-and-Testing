#/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_WilliamsFractal_1wTrend_PriceAction"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Williams Fractal on 1D (needs 2-bar confirmation)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Bearish fractal: high[n-2] < high[n-1] > high[n] and high[n-1] > high[n-3] and high[n-1] > high[n+1]
    # Bullish fractal: low[n-2] > low[n-1] < low[n] and low[n-1] < low[n-3] and low[n-1] < low[n+1]
    bearish = np.zeros(len(high_1d), dtype=bool)
    bullish = np.zeros(len(low_1d), dtype=bool)
    
    for i in range(2, len(high_1d)-2):
        if (high_1d[i-2] < high_1d[i-1] and 
            high_1d[i] < high_1d[i-1] and
            high_1d[i-3] < high_1d[i-1] and
            high_1d[i+1] < high_1d[i-1]):
            bearish[i-1] = True
        if (low_1d[i-2] > low_1d[i-1] and 
            low_1d[i] > low_1d[i-1] and
            low_1d[i-3] > low_1d[i-1] and
            low_1d[i+1] > low_1d[i-1]):
            bullish[i-1] = True
    
    # Add 2-bar delay for confirmation
    bearish_1d = align_htf_to_ltf(prices, df_1d, bearish.astype(float), additional_delay_bars=2)
    bullish_1d = align_htf_to_ltf(prices, df_1d, bullish.astype(float), additional_delay_bars=2)
    
    # Weekly trend: EMA34 on 1W
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    ema34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Price action: price > 12-period high for momentum
    high_12 = pd.Series(high).rolling(window=12, min_periods=12).max().values
    low_12 = pd.Series(low).rolling(window=12, min_periods=12).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if required data unavailable
        if np.isnan(bearish_1d[i]) or np.isnan(bullish_1d[i]) or \
           np.isnan(ema34_1w_aligned[i]) or np.isnan(high_12[i]) or np.isnan(low_12[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: bullish fractal + price above weekly EMA + price making new high
            if (bullish_1d[i] > 0.5 and 
                price > ema34_1w_aligned[i] and 
                price > high_12[i-1]):
                signals[i] = 0.25
                position = 1
            # Short: bearish fractal + price below weekly EMA + price making new low
            elif (bearish_1d[i] > 0.5 and 
                  price < ema34_1w_aligned[i] and 
                  price < low_12[i-1]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: bearish fractal or price breaks weekly trend
            if (bearish_1d[i] > 0.5 or 
                price < ema34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: bullish fractal or price breaks weekly trend
            if (bullish_1d[i] > 0.5 or 
                price > ema34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals