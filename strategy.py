#!/usr/bin/env python3
"""
Hypothesis: 12h KAMA trend + RSI mean reversion + volume confirmation + choppiness regime filter.
Long when KAMA trending up, RSI < 30 (oversold), volume > 1.5x average, and choppy market (CHOP > 61.8).
Short when KAMA trending down, RSI > 70 (overbought), volume > 1.5x average, and choppy market (CHOP > 61.8).
Exit on opposite RSI extreme or KAMA trend reversal.
KAMA adapts to market noise, RSI captures mean reversion in ranging markets, volume confirms legitimacy,
and chop filter ensures we only mean revert in ranging conditions. Designed for 12h timeframe targeting
50-150 total trades over 4 years with low frequency to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate KAMA (primary timeframe)
    # Efficiency Ratio (ER)
    change = np.abs(np.diff(close, n=10))  # 10-period change
    volatility = np.sum(np.abs(np.diff(close)), axis=1)  # 10-period volatility
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    # Initialize KAMA
    kama = np.full(n, np.nan)
    kama[9] = close[9]  # start after 10 periods
    for i in range(10, n):
        if not np.isnan(sc[i]) and not np.isnan(kama[i-1]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # Calculate RSI (14-period)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    # Handle first values
    rsi[:13] = np.nan
    
    # Volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Choppiness Index (14-period) - primary timeframe
    # True Range
    tr1 = np.abs(high[1:] - low[1:])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])  # align with close
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    # Max/min close over 14 periods
    max_close = pd.Series(close).rolling(window=14, min_periods=14).max().values
    min_close = pd.Series(close).rolling(window=14, min_periods=14).min().values
    # Choppiness Index
    chop = np.where(
        (atr > 0) & (max_close != min_close),
        100 * np.log10(np.sum(atr) / (max_close - min_close)) / np.log10(14),
        50
    )
    # For rolling sum of ATR, we need to calculate it properly
    atr_sum = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    chop = np.where(
        (atr_sum > 0) & (max_close != min_close),
        100 * np.log10(atr_sum / (max_close - min_close)) / np.log10(14),
        50
    )
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(chop[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        kama_val = kama[i]
        rsi_val = rsi[i]
        vol_ma_val = vol_ma[i]
        chop_val = chop[i]
        price = close[i]
        vol_current = volume[i]
        
        if position == 0:
            # Long: KAMA trending up, RSI oversold, volume spike, choppy market
            if (price > kama_val and rsi_val < 30 and vol_current > 1.5 * vol_ma_val and chop_val > 61.8):
                signals[i] = 0.25
                position = 1
            # Short: KAMA trending down, RSI overbought, volume spike, choppy market
            elif (price < kama_val and rsi_val > 70 and vol_current > 1.5 * vol_ma_val and chop_val > 61.8):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: RSI overbought OR KAMA trend down
                if (rsi_val > 70 or price < kama_val):
                    exit_signal = True
            else:  # position == -1
                # Exit short: RSI oversold OR KAMA trend up
                if (rsi_val < 30 or price > kama_val):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_KAMA_RSI_Volume_Chop"
timeframe = "12h"
leverage = 1.0