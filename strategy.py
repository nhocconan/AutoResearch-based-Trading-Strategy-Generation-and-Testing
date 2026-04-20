#!/usr/bin/env python3
# 4h_12h_KAMA_RSI_TrendFilter
# Hypothesis: KAMA trend direction + RSI pullback with volume confirmation on 4h. KAMA adapts to market noise, reducing whipsaw in ranging markets. RSI (14) < 40 for long, > 60 for short during pullbacks. Volume > 1.5x average confirms momentum. Trend filter uses 12h EMA34 to avoid counter-trend trades. Designed for fewer trades (~20-40/year) to minimize fee drag and work in both bull/bear via trend alignment.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_KAMA_RSI_TrendFilter"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 12h data ONCE before loop for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # === 12h EMA34 for trend filter ===
    close_12h = df_12h['close'].values
    close_12h_series = pd.Series(close_12h)
    ema34_12h = close_12h_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    
    # === 4h: KAMA trend direction ===
    close = prices['close'].values
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close, n=10))
    volatility = np.sum(np.abs(np.diff(close, n=1)), axis=0)
    # Pad arrays to match length
    change = np.concatenate([np.full(10, np.nan), change])
    volatility = np.concatenate([np.full(1, np.nan), volatility[:-1]])  # align with change
    er = np.where(volatility > 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    # KAMA calculation
    kama = np.full(n, np.nan)
    kama[0] = close[0]
    for i in range(1, n):
        if np.isnan(sc[i]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    kama_dir = np.where(close > kama, 1, -1)  # 1: above KAMA (up trend), -1: below KAMA (down trend)
    
    # === 4h: RSI(14) ===
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    # First average
    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)
    avg_gain[14] = np.mean(gain[1:15])
    avg_loss[14] = np.mean(loss[1:15])
    # Wilder smoothing
    for i in range(15, n):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    rs = np.where(avg_loss > 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    
    # === 4h: Volume ratio (current vs 20-period average) ===
    volume = prices['volume'].values
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(34, n):  # Start after EMA and KAMA/RSI warmup
        # Get values
        close_val = close[i]
        kama_dir_val = kama_dir[i]
        rsi_val = rsi[i]
        ema34_12h_val = ema34_12h_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if (np.isnan(kama_dir_val) or np.isnan(rsi_val) or np.isnan(ema34_12h_val) or 
            np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price above KAMA (uptrend), RSI < 40 (pullback), volume confirmation, and above 12h EMA34
            if (kama_dir_val == 1 and rsi_val < 40 and vol_ratio_val > 1.5 and 
                close_val > ema34_12h_val):
                signals[i] = 0.25
                position = 1
            # Short: Price below KAMA (downtrend), RSI > 60 (pullback), volume confirmation, and below 12h EMA34
            elif (kama_dir_val == -1 and rsi_val > 60 and vol_ratio_val > 1.5 and 
                  close_val < ema34_12h_val):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price crosses below KAMA or RSI > 70 (overbought)
            if kama_dir_val == -1 or rsi_val > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price crosses above KAMA or RSI < 30 (oversold)
            if kama_dir_val == 1 or rsi_val < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals