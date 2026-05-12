#!/usr/bin/env python3
"""
6h_Stochastic_Divergence_1dTrend_Confirmation
Hypothesis: Stochastic oscillator (14,3,3) identifies overbought/oversold conditions on 6h chart.
Divergence between price and Stochastic (bullish: price makes lower low, Stoch makes higher low;
bearish: price makes higher high, Stoch makes lower high) signals potential reversals.
Trades are only taken in the direction of the 1d EMA50 trend to avoid counter-trend whipsaws.
Volume confirmation (>1.3x average) filters low-momentum signals.
Works in bull/bear by following 1d trend direction, reducing false signals in ranging markets.
"""

name = "6h_Stochastic_Divergence_1dTrend_Confirmation"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA50 trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Stochastic oscillator (14,3,3) on 6h
    # %K = (Current Close - Lowest Low) / (Highest High - Lowest Low) * 100
    # %D = SMA of %K, period 3
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    k = 100 * (close - lowest_low) / (highest_high - lowest_low)
    d = pd.Series(k).rolling(window=3, min_periods=3).mean().values
    
    # Volatility filter: average true range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume spike: >1.3x 20-period average (6h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.3 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after EMA50 and Stochastic warmup
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(k[i]) or np.isnan(d[i]) or 
            np.isnan(atr[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Bullish divergence: price makes lower low, Stoch makes higher low
            bullish_div = (low[i] < low[i-1] and 
                          k[i] > d[i] and 
                          k[i] < 30 and  # Oversold
                          low[i] < low[i-2] and 
                          k[i] > k[i-2])
            
            # Bearish divergence: price makes higher high, Stoch makes lower high
            bearish_div = (high[i] > high[i-1] and 
                          k[i] < d[i] and 
                          k[i] > 70 and  # Overbought
                          high[i] > high[i-2] and 
                          k[i] < k[i-2])
            
            # LONG: Bullish divergence + 1d EMA50 uptrend + volume spike
            if (bullish_div and 
                close[i] > ema_50_1d_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Bearish divergence + 1d EMA50 downtrend + volume spike
            elif (bearish_div and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Bearish divergence or price crosses below EMA50
            if (bearish_div or close[i] < ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Bullish divergence or price crosses above EMA50
            if (bullish_div or close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals