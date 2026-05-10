#!/usr/bin/env python3
"""
1d_KAMA_Trend_Filter_With_RSI_And_Chop_Filter
Hypothesis: KAMA trend direction filters RSI extremes with chop filter to avoid whipsaw.
Uses weekly timeframe for regime detection to work in both bull and bear markets.
Target: 10-20 trades/year per symbol with strict entry conditions to minimize fee drag.
"""

name = "1d_KAMA_Trend_Filter_With_RSI_And_Chop_Filter"
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
    
    # Calculate KAMA(10) for trend
    def kama(close, period=10):
        change = np.abs(np.diff(close, n=period))
        volatility = np.sum(np.abs(np.diff(close)), axis=1)
        er = np.where(volatility != 0, change / volatility, 0)
        sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
        kama = np.full_like(close, np.nan)
        kama[period] = close[period]
        for i in range(period+1, len(close)):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        return kama
    
    kama_val = kama(close, 10)
    
    # Calculate RSI(14)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.full_like(close, np.nan)
    avg_loss = np.full_like(close, np.nan)
    for i in range(14, len(close)):
        if i == 14:
            avg_gain[i] = np.mean(gain[1:15])
            avg_loss[i] = np.mean(loss[1:15])
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Chop filter using weekly data
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range for chop calculation
    tr1 = high_1w - low_1w
    tr2 = np.abs(np.roll(high_1w, 1) - close_1w)
    tr3 = np.abs(np.roll(low_1w, 1) - close_1w)
    tr2[0] = np.inf
    tr3[0] = np.inf
    tr_1w = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Chop calculation: (sum of TR / (max(high) - min(low))) * 100
    chop = np.full(len(close_1w), np.nan)
    for i in range(14, len(close_1w)):
        atr_sum = np.sum(tr_1w[i-14:i+1])
        max_high = np.max(high_1w[i-14:i+1])
        min_low = np.min(low_1w[i-14:i+1])
        range_val = max_high - min_low
        if range_val != 0:
            chop[i] = (atr_sum / range_val) * 100
        else:
            chop[i] = 50
    
    chop_aligned = align_htf_to_ltf(prices, df_1w, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(15, 15)  # Ensure RSI and KAMA are ready
    
    for i in range(start_idx, n):
        if np.isnan(kama_val[i]) or np.isnan(rsi[i]) or np.isnan(chop_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Chop filter: only trade when market is trending (CHOP < 38.2) or ranging (CHOP > 61.8)
        # In trending markets: follow KAMA direction
        # In ranging markets: fade extreme RSI
        if chop_aligned[i] < 38.2:  # Trending market
            if position == 0:
                # Go with KAMA trend
                if close[i] > kama_val[i]:
                    signals[i] = 0.25
                    position = 1
                elif close[i] < kama_val[i]:
                    signals[i] = -0.25
                    position = -1
            elif position == 1:
                # Exit long when price crosses below KAMA
                if close[i] < kama_val[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short when price crosses above KAMA
                if close[i] > kama_val[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
        elif chop_aligned[i] > 61.8:  # Ranging market
            if position == 0:
                # Fade RSI extremes
                if rsi[i] < 30:  # Oversold - go long
                    signals[i] = 0.25
                    position = 1
                elif rsi[i] > 70:  # Overbought - go short
                    signals[i] = -0.25
                    position = -1
            elif position == 1:
                # Exit long when RSI returns to neutral
                if rsi[i] > 50:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short when RSI returns to neutral
                if rsi[i] < 50:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
        else:  # Neutral chop - stay flat
            if position != 0:
                signals[i] = 0.0
                position = 0
    
    return signals