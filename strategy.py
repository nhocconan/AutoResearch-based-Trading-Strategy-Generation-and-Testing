#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_KAMA_RSI_Trend_V1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === 1d: KAMA trend filter ===
    close_1d = df_1d['close'].values
    # Calculate Efficiency Ratio (ER)
    change = np.abs(np.diff(close_1d, k=10, prepend=close_1d[:10]))
    volatility = np.sum(np.abs(np.diff(close_1d, k=1, prepend=close_1d[0])), axis=0)
    er = change / np.where(volatility > 0, volatility, 1.0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    kama_1d = kama
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    
    # === 1d: RSI(14) filter ===
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_loss / np.where(avg_gain > 0, avg_gain, np.nan)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # === 4h: Price and volume ===
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume ratio (current vs 20-period average)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Get values
        close_val = close[i]
        kama_val = kama_1d_aligned[i]
        rsi_val = rsi_1d_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if np.isnan(kama_val) or np.isnan(rsi_val) or np.isnan(vol_ratio_val):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price above KAMA + RSI not overbought + volume confirmation
            if (close_val > kama_val and
                rsi_val < 70 and
                vol_ratio_val > 1.3):
                signals[i] = 0.25
                position = 1
            # Short: Price below KAMA + RSI not oversold + volume confirmation
            elif (close_val < kama_val and
                  rsi_val > 30 and
                  vol_ratio_val > 1.3):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price below KAMA or RSI overbought or low volume
            if (close_val < kama_val or
                rsi_val > 75 or
                vol_ratio_val < 0.9):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price above KAMA or RSI oversold or low volume
            if (close_val > kama_val or
                rsi_val < 25 or
                vol_ratio_val < 0.9):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals