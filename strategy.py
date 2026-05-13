#!/usr/bin/env python3
"""
1d_WMA_Trend_With_Volume_Filter
Hypothesis: Weighted Moving Average (WMA) with recent price emphasis provides responsive yet smooth trend signals.
Combined with volume confirmation to ensure institutional participation, this strategy captures trends in both bull and bear markets.
Uses weekly trend filter for higher timeframe confirmation to reduce false signals.
Designed for low trade frequency (target: 10-25 trades/year) to minimize fee drain on daily timeframe.
"""

name = "1d_WMA_Trend_With_Volume_Filter"
timeframe = "1d"
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
    
    # Get weekly data for trend filter (once before loop)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA50 for trend filter
    ema50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate daily WMA20: weighted average with more weight on recent prices
    weights = np.arange(1, 21)
    wma = np.convolve(close, weights[::-1], mode='full')[:len(close)] / weights.sum()
    wma[:19] = np.nan  # Not enough data for full window
    
    # Volume confirmation: current volume > 1.5x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        if np.isnan(wma[i]):
            signals[i] = 0.0
            continue
            
        if position == 0:
            # LONG: Price above WMA20 with volume spike and above weekly EMA50
            if (close[i] > wma[i] and 
                volume_spike[i] and 
                close[i] > trend_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price below WMA20 with volume spike and below weekly EMA50
            elif (close[i] < wma[i] and 
                  volume_spike[i] and 
                  close[i] < trend_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below WMA20 or weekly trend turns bearish
            if (close[i] < wma[i] or 
                close[i] < trend_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above WMA20 or weekly trend turns bullish
            if (close[i] > wma[i] or 
                close[i] > trend_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals