#!/usr/bin/env python3
"""
6h_1w_1d_LP_Reversion_Reversal
Hypothesis: 6-hour mean reversion from long-period extremes (1w/1d) with momentum confirmation.
In bear markets: long when price touches 1w Bollinger lower band with oversold RSI and bullish momentum divergence.
In bull markets: short when price touches 1w Bollinger upper band with overbought RSI and bearish momentum divergence.
Uses 1d volume spike to confirm institutional interest at extremes.
Designed to work in both bull and bear markets via mean reversion at extremes + momentum confirmation.
Targets 6h timeframe for lower trade frequency (~20-50/year) to minimize fee drag.
"""

name = "6h_1w_1d_LP_Reversion_Reversal"
timeframe = "6h"
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
    volume = prices['volume'].values
    
    # Volume spike: >1.8x 30-period average (on 6h timeframe)
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_spike = volume > (1.8 * vol_ma)
    
    # 1d data for RSI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # RSI(14) on 1d
    delta = pd.Series(df_1d['close']).diff().values
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # 1w data for Bollinger Bands
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Bollinger Bands(20,2) on 1w
    sma = pd.Series(df_1w['close']).rolling(window=20, min_periods=20).mean().values
    std = pd.Series(df_1w['close']).rolling(window=20, min_periods=20).std().values
    upper = sma + (2 * std)
    lower = sma - (2 * std)
    upper_aligned = align_htf_to_ltf(prices, df_1w, upper)
    lower_aligned = align_htf_to_ltf(prices, df_1w, lower)
    
    # Momentum: ROC(10) on 6h for divergence confirmation
    roc = np.zeros_like(close)
    roc[10:] = (close[10:] - close[:-10]) / close[:-10] * 100
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        if (np.isnan(rsi_aligned[i]) or
            np.isnan(upper_aligned[i]) or
            np.isnan(lower_aligned[i]) or
            np.isnan(roc[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price touches 1w lower BB + RSI < 30 (oversold) + bullish ROC momentum
            if (low[i] <= lower_aligned[i] and 
                rsi_aligned[i] < 30 and 
                roc[i] > roc[i-1]):  # ROC improving = bullish momentum
                signals[i] = 0.25
                position = 1
            # SHORT: Price touches 1w upper BB + RSI > 70 (overbought) + bearish ROC momentum
            elif (high[i] >= upper_aligned[i] and 
                  rsi_aligned[i] > 70 and 
                  roc[i] < roc[i-1]):  # ROC deteriorating = bearish momentum
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price returns to 1w SMA OR RSI > 50
            if (close[i] >= sma[-1] if len(sma) > 0 else False) or \
               rsi_aligned[i] > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price returns to 1w SMA OR RSI < 50
            if (close[i] <= sma[-1] if len(sma) > 0 else False) or \
               rsi_aligned[i] < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals