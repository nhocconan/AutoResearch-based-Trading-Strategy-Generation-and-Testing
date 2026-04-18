#!/usr/bin/env python3
"""
4h KAMA Direction + RSI + Chop Filter with 200 EMA Trend Filter
KAMA adapts to market efficiency, RSI filters extremes, Chop filter avoids whipsaws in range markets.
200 EMA ensures alignment with major trend. Designed for low trade frequency (<30/year) with
strong performance in both bull and bear markets by avoiding false signals during consolidation.
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
    
    # KAMA calculation (ER=10, fast=2, slow=30)
    change = np.abs(np.diff(close, k=10))
    volatility = np.sum(np.abs(np.diff(close)), axis=0)
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama = np.full_like(close, np.nan, dtype=float)
    kama[9] = close[9]  # seed
    for i in range(10, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI(14)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # 200 EMA trend filter
    ema_200 = pd.Series(close).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Chopiness Index (14-period)
    atr = pd.Series(np.maximum(np.maximum(high - low, np.abs(high - np.roll(close, 1))),
                               np.abs(low - np.roll(close, 1)))).rolling(window=14, min_periods=14).mean().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10((highest_high - lowest_low) / (np.sum(atr, axis=1, dtype=object) if hasattr(np.sum(atr), 'dtype') else np.nansum(atr.reshape(-1, 14), axis=1)) / np.log(14))
    # Fix chop calculation
    chop = np.full_like(close, np.nan)
    for i in range(13, n):
        atr_sum = np.sum(np.maximum(np.maximum(high[i-13:i+1] - low[i-13:i+1], 
                                              np.abs(high[i-13:i+1] - np.roll(close[i-13:i+1], 1))),
                                  np.abs(low[i-13:i+1] - np.roll(close[i-13:i+1], 1))))
        hh = np.max(high[i-13:i+1])
        ll = np.min(low[i-13:i+1])
        if atr_sum > 0:
            chop[i] = 100 * np.log10((hh - ll) / atr_sum) / np.log(14)
        else:
            chop[i] = 50
    
    # Volume confirmation (1.5x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 100  # need enough history for all indicators
    
    for i in range(start_idx, n):
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(ema_200[i]) or 
            np.isnan(chop[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        kama_val = kama[i]
        rsi_val = rsi[i]
        ema200_val = ema_200[i]
        chop_val = chop[i]
        vol_ok = volume_ok[i]
        
        if position == 0:
            # Long: KAMA up, RSI not overbought, chop < 61.8 (trending), above EMA200, volume
            if (price > kama_val and 
                rsi_val < 70 and 
                chop_val < 61.8 and 
                price > ema200_val and 
                vol_ok):
                signals[i] = 0.25
                position = 1
            # Short: KAMA down, RSI not oversold, chop < 61.8, below EMA200, volume
            elif (price < kama_val and 
                  rsi_val > 30 and 
                  chop_val < 61.8 and 
                  price < ema200_val and 
                  vol_ok):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long position: hold until KAMA turns down or RSI overbought
            signals[i] = 0.25
            if price < kama_val or rsi_val > 80:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            # Short position: hold until KAMA turns up or RSI oversold
            signals[i] = -0.25
            if price > kama_val or rsi_val < 20:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_KAMA_RSI_Chop_EMA200"
timeframe = "4h"
leverage = 1.0