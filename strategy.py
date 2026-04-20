#!/usr/bin/env python3
# 12h_1d_KAMA_RSI_With_Volume_Confirmation
# Hypothesis: 12h KAMA direction + RSI momentum + volume confirmation for trend following.
# Uses 1d trend filter (price > 200 EMA) to avoid counter-trend trades.
# Works in bull/bear via 1d trend filter - only trade with the daily trend.
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_KAMA_RSI_With_Volume_Confirmation"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Get daily data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    # === Calculate 1d EMA200 for trend filter ===
    close_1d = df_1d['close'].values
    close_1d_series = pd.Series(close_1d)
    ema200_1d = close_1d_series.ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # === 12h: KAMA calculation ===
    close = prices['close'].values
    direction = np.abs(np.diff(close, n=10))  # 10-period net change
    volatility = np.sum(np.abs(np.diff(close, n=1)), axis=0) if len(close) > 1 else 0
    # Full calculation: ER = direction / volatility, but we'll use simplified version
    # For efficiency, we'll use a basic adaptive approach
    change = np.abs(np.diff(close, n=1))
    volatility_sum = np.convolve(change, np.ones(10), mode='same')
    er = np.where(volatility_sum > 0, direction / volatility_sum, 0)
    sc = (er * (0.6645 - 0.0645) + 0.0645) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # === 12h: RSI(14) ===
    delta = np.diff(close, n=1)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss > 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # === 12h: Volume ratio (current vs 20-period average) ===
    volume = prices['volume'].values
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    # Align daily EMA200 to 12h
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):  # Start after EMA200 warmup
        # Get values
        close_val = close[i]
        kama_val = kama[i]
        rsi_val = rsi[i]
        vol_ratio_val = vol_ratio[i]
        ema200_1d_val = ema200_1d_aligned[i]
        
        # Skip if any value is NaN
        if (np.isnan(kama_val) or np.isnan(rsi_val) or np.isnan(vol_ratio_val) or 
            np.isnan(ema200_1d_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price above KAMA (bullish momentum) + RSI > 50 + volume confirmation + above daily EMA200
            if (close_val > kama_val and 
                rsi_val > 50 and 
                vol_ratio_val > 1.5 and 
                close_val > ema200_1d_val):
                signals[i] = 0.25
                position = 1
            # Short: Price below KAMA (bearish momentum) + RSI < 50 + volume confirmation + below daily EMA200
            elif (close_val < kama_val and 
                  rsi_val < 50 and 
                  vol_ratio_val > 1.5 and 
                  close_val < ema200_1d_val):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price crosses below KAMA or RSI < 40
            if close_val < kama_val or rsi_val < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price crosses above KAMA or RSI > 60
            if close_val > kama_val or rsi_val > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals