#!/usr/bin/env python3
# 12h_1D_KAMA_Trend_RSI_Filter
# Hypothesis: 12-hour KAMA trend with daily RSI filter and volume confirmation.
# Uses KAMA to identify trend direction, RSI for momentum exhaustion (RSI > 70 for short, RSI < 30 for long),
# and volume spike for confirmation. Designed for fewer trades (target 15-30/year) to avoid fee drag.
# Works in bull markets via trend following and in bear markets via mean reversion at extremes.

name = "12h_1D_KAMA_Trend_RSI_Filter"
timeframe = "12h"
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
    
    # Volume spike: >2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # Daily data for RSI and KAMA trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Daily RSI(14)
    delta = pd.Series(df_1d['close']).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi_14 = 100 - (100 / (1 + rs))
    rsi_14 = rsi_14.fillna(50).values  # neutral when undefined
    
    # Daily KAMA for trend
    close_1d = df_1d['close']
    # Efficiency Ratio
    change = abs(close_1d - close_1d.shift(10))
    volatility = abs(close_1d.diff()).rolling(window=10, min_periods=10).sum()
    er = change / volatility.replace(0, np.nan)
    er = er.fillna(0).values
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    # KAMA
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d.iloc[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d.iloc[i] - kama[i-1])
    kama = kama
    
    # Align daily indicators to 12h timeframe
    rsi_14_aligned = align_htf_to_ltf(prices, df_1d, rsi_14)
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        if (np.isnan(rsi_14_aligned[i]) or 
            np.isnan(kama_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price above KAMA (uptrend) + RSI < 30 (oversold) + volume spike
            if (close[i] > kama_aligned[i] and 
                rsi_14_aligned[i] < 30 and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price below KAMA (downtrend) + RSI > 70 (overbought) + volume spike
            elif (close[i] < kama_aligned[i] and 
                  rsi_14_aligned[i] > 70 and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below KAMA OR RSI > 70 (overbought)
            if close[i] < kama_aligned[i] or rsi_14_aligned[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above KAMA OR RSI < 30 (oversold)
            if close[i] > kama_aligned[i] or rsi_14_aligned[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals