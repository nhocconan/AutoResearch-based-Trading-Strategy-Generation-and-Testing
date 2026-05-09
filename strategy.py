#!/usr/bin/env python3
# Hypothesis: 1d Kamas + RSI with Chop regime filter
# Uses KAMA(14) trend direction on 1d timeframe, filtered by RSI(14) and Choppiness Index(14)
# KAMA adapts to market noise, reducing whipsaw in choppy markets
# RSI(14) provides mean-reversion signals within trend context
# Choppiness Index determines market regime: >61.8 = range (favor mean reversion), <38.2 = trend (favor trend following)
# In ranging markets: long when RSI<30, short when RSI>70
# In trending markets: long when KAMA rising and RSI>50, short when KAMA falling and RSI<50
# Position size: 0.25 for clear signals, 0.125 for weaker signals
# Target: 15-25 trades per year with controlled frequency

name = "1d_KAMA_RSI_ChopRegime"
timeframe = "1d"
leverage = 1.0

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
    
    # Get weekly data for Chop regime filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    # Calculate Choppiness Index on weekly data
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w[1:] - low_1w[1:]
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with index
    
    # ATR(14) - sum of TR over 14 periods
    atr_14 = np.zeros_like(tr)
    for i in range(14, len(tr)):
        atr_14[i] = np.nansum(tr[i-13:i+1])
    
    # Sum of absolute price changes over 14 periods
    price_change = np.abs(np.diff(close_1w, prepend=close_1w[0]))
    sum_price_change_14 = np.zeros_like(price_change)
    for i in range(14, len(price_change)):
        sum_price_change_14[i] = np.nansum(price_change[i-13:i+1])
    
    # Chop = 100 * log10(sum_price_change / atr) / log10(14)
    chop = np.zeros_like(close_1w)
    mask = (atr_14 > 0) & (sum_price_change_14 > 0)
    chop[mask] = 100 * np.log10(sum_price_change_14[mask] / atr_14[mask]) / np.log10(14)
    chop = np.concatenate([[np.nan] * 13, chop[13:]])  # first 14 values NaN
    
    # Align Chop to daily timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1w, chop)
    
    # Get daily data for KAMA and RSI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # KAMA(14) - Kaufman Adaptive Moving Average
    # Efficiency Ratio
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.zeros_like(close_1d)
    for i in range(1, len(close_1d)):
        volatility[i] = volatility[i-1] + np.abs(close_1d[i] - close_1d[i-1])
    
    er = np.zeros_like(close_1d)
    mask = volatility > 0
    er[mask] = change[mask] / volatility[mask]
    
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    
    # KAMA calculation
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # RSI(14)
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    for i in range(14, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.zeros_like(avg_gain)
    mask = avg_loss != 0
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rsi = 100 - (100 / (1 + rs))
    rsi[0:14] = 50  # neutral before enough data
    
    # Align indicators to lower timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(chop_aligned[i]) or np.isnan(kama_aligned[i]) or 
            np.isnan(rsi_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        chop_val = chop_aligned[i]
        kama_val = kama_aligned[i]
        rsi_val = rsi_aligned[i]
        kama_prev = kama_aligned[i-1] if i > 0 else kama_val
        
        if position == 0:
            # Determine market regime
            if chop_val > 61.8:  # Ranging market
                # Mean reversion: buy oversold, sell overbought
                if rsi_val < 30:
                    signals[i] = 0.25
                    position = 1
                elif rsi_val > 70:
                    signals[i] = -0.25
                    position = -1
            elif chop_val < 38.2:  # Trending market
                # Trend following: follow KAMA direction with RSI filter
                if kama_val > kama_prev and rsi_val > 50:
                    signals[i] = 0.25
                    position = 1
                elif kama_val < kama_prev and rsi_val < 50:
                    signals[i] = -0.25
                    position = -1
            else:  # Transition zone - reduced size
                if rsi_val < 35:
                    signals[i] = 0.125
                    position = 1
                elif rsi_val > 65:
                    signals[i] = -0.125
                    position = -1
        
        elif position == 1:
            # Exit long conditions
            if chop_val > 61.8:  # In range, exit at RSI extremes
                if rsi_val > 70:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif chop_val < 38.2:  # In trend, exit on trend reversal
                if kama_val < kama_prev or rsi_val < 40:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # Transition zone
                if rsi_val > 60:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.125 if position == 1 else -0.125
        
        elif position == -1:
            # Exit short conditions
            if chop_val > 61.8:  # In range, exit at RSI extremes
                if rsi_val < 30:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            elif chop_val < 38.2:  # In trend, exit on trend reversal
                if kama_val > kama_prev or rsi_val > 60:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:  # Transition zone
                if rsi_val < 40:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.125 if position == 1 else -0.125
    
    return signals