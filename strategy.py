#!/usr/bin/env python3
"""
4h_RSI_Stochastic_Combo_v1
Hypothesis: Combines RSI(14) for overbought/oversold conditions with Stochastic(14,3,3)
for momentum confirmation on 4h timeframe. Uses 1-day trend filter to align with higher
timeframe direction. Designed for low trade frequency (target: 20-40 trades/year) to
minimize fee drag while capturing reversals in both bull and bear markets.
"""

name = "4h_RSI_Stochastic_Combo_v1"
timeframe = "4h"
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
    
    # RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Stochastic(14,3,3)
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    k_percent = np.divide((close - lowest_low) * 100, (highest_high - lowest_low), 
                          out=np.zeros_like(close), where=(highest_high - lowest_low)!=0)
    d_percent = pd.Series(k_percent).rolling(window=3, min_periods=3).mean().values
    
    # 1-day trend filter: EMA of daily close
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volume confirmation: current volume > 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if any critical value is NaN
        if (np.isnan(rsi[i]) or np.isnan(d_percent[i]) or 
            np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: RSI < 30 (oversold) AND Stochastic %D < 20 AND price > 1-day EMA with volume confirmation
            if rsi[i] < 30 and d_percent[i] < 20 and close[i] > ema_1d_aligned[i] and volume[i] > vol_ma[i]:
                signals[i] = 0.25
                position = 1
            # Short: RSI > 70 (overbought) AND Stochastic %D > 80 AND price < 1-day EMA with volume confirmation
            elif rsi[i] > 70 and d_percent[i] > 80 and close[i] < ema_1d_aligned[i] and volume[i] > vol_ma[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: RSI > 50 (momentum shift) OR price crosses below 1-day EMA
            if rsi[i] > 50 or close[i] < ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: RSI < 50 (momentum shift) OR price crosses above 1-day EMA
            if rsi[i] < 50 or close[i] > ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals