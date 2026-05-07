#!/usr/bin/env python3
name = "4h_KAMA_RSI_TrendFollow"
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
    
    # Load 1d data ONCE before loop for KAMA and RSI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 1d KAMA: Efficiency Ratio (ER) period 10, Fast SC 2, Slow SC 30
    close_1d = df_1d['close']
    change = abs(close_1d.diff(10))
    volatility = close_1d.diff().abs().rolling(10).sum()
    er = change / volatility.replace(0, np.nan)
    sc = (er * (2/10 - 1/30) + 1/30) ** 2
    kama = [np.nan] * len(close_1d)
    kama[0] = close_1d.iloc[0]
    for i in range(1, len(close_1d)):
        if np.isnan(sc.iloc[i]) or np.isnan(kama[i-1]):
            kama[i] = np.nan
        else:
            kama[i] = kama[i-1] + sc.iloc[i] * (close_1d.iloc[i] - kama[i-1])
    kama = np.array(kama)
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    # 1d RSI(14)
    delta = close_1d.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # Volume filter: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > 1.5 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Wait for indicators
    
    for i in range(start_idx, n):
        if np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price above KAMA and RSI > 50 with volume
            if (close[i] > kama_aligned[i] and rsi_aligned[i] > 50 and vol_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price below KAMA and RSI < 50 with volume
            elif (close[i] < kama_aligned[i] and rsi_aligned[i] < 50 and vol_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Price below KAMA or RSI < 50
            if close[i] < kama_aligned[i] or rsi_aligned[i] < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Price above KAMA or RSI > 50
            if close[i] > kama_aligned[i] or rsi_aligned[i] > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 4h KAMA with RSI(14) trend filter and volume confirmation.
# KAMA adapts to market noise, reducing whipsaw in choppy markets while
# following trends in strong moves. RSI confirms momentum direction.
# Volume ensures institutional participation. Position size 0.25 limits drawdown.
# Works in both bull (trend following) and bear (adaptive filtering reduces false signals).
# Target: ~20-30 trades/year to avoid fee drag.