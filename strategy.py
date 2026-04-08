#!/usr/bin/env python3
"""
4h_metalawrence_rsi_bands_v1
Hypothesis: Use RSI(14) with dynamic Bollinger Bands (20,2) on RSI to identify extreme mean-reversion zones.
Only trade in direction of 1d trend (price above/below EMA50). Enter when RSI touches upper/lower band and shows rejection.
Exit on opposite RSI band touch or trend reversal.
Designed for low trade frequency (<30/year) with clear entry/exit rules to avoid overtrading.
Works in both bull/bear via trend filter and mean-reversion logic.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_metalawrence_rsi_bands_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Bollinger Bands on RSI (20, 2)
    rsi_series = pd.Series(rsi)
    rsi_ma = rsi_series.rolling(window=20, min_periods=20).mean()
    rsi_std = rsi_series.rolling(window=20, min_periods=20).std()
    rsi_upper = rsi_ma + 2 * rsi_std
    rsi_lower = rsi_ma - 2 * rsi_std
    
    # 1d trend filter: EMA50
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    ema_50 = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Wait for BB warmup
        # Skip if data not ready
        if (np.isnan(rsi[i]) or np.isnan(rsi_upper[i]) or np.isnan(rsi_lower[i]) or 
            np.isnan(ema_50_aligned[i])):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long
            # Exit: RSI touches upper band or trend turns bearish
            if rsi[i] >= rsi_upper[i] or close[i] < ema_50_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short
            # Exit: RSI touches lower band or trend turns bullish
            if rsi[i] <= rsi_lower[i] or close[i] > ema_50_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Long: RSI touches lower band with rejection + bullish trend
            if (rsi[i] <= rsi_lower[i] and 
                close[i] > ema_50_aligned[i] and 
                rsi[i] > rsi[i-1]):  # RSI rising off lower band
                position = 1
                signals[i] = 0.25
            # Short: RSI touches upper band with rejection + bearish trend
            elif (rsi[i] >= rsi_upper[i] and 
                  close[i] < ema_50_aligned[i] and 
                  rsi[i] < rsi[i-1]):  # RSI falling off upper band
                position = -1
                signals[i] = -0.25
    
    return signals