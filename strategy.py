#!/usr/bin/env python3
# Hypothesis: 12h KAMA (Kaufman Adaptive Moving Average) with RSI and Choppiness Index filter
# Long when KAMA trending up, RSI > 50, and Chop < 40 (trending market)
# Short when KAMA trending down, RSI < 50, and Chop < 40 (trending market)
# Exit when RSI crosses 50 or Chop > 50 (range market)
# Uses 1d trend filter for higher timeframe confirmation
# Designed to capture trending moves with low frequency to minimize fee drag
# Target: 50-120 total trades over 4 years (12-30/year) with size 0.25

name = "12h_KAMA_RSI_Chop_Trend"
timeframe = "12h"
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
    
    # Calculate 1d KAMA for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d KAMA (Kaufman Adaptive Moving Average)
    close_1d = df_1d['close'].values
    # Efficiency Ratio
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.abs(np.diff(close_1d))
    er = np.divide(change, volatility, out=np.zeros_like(change), where=volatility!=0)
    # Smoothing Constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    # KAMA calculation
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # Align 1d KAMA to 12h timeframe
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    # Calculate 12h RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate 12h Choppiness Index(14)
    atr = np.zeros(n)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.insert(tr, 0, high[0] - low[0])  # first TR
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean()
    
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max()
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min()
    
    chop = np.where(
        (max_high - min_low) != 0,
        100 * np.log10(atr.rolling(window=14, min_periods=14).sum() / (max_high - min_low)) / np.log10(14),
        50
    )
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(kama_1d_aligned[i]) or np.isnan(rsi[i]) or np.isnan(chop[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: KAMA up, RSI > 50, Chop < 40 (trending up)
            if (kama_1d_aligned[i] > kama_1d_aligned[i-1] and 
                rsi[i] > 50 and 
                chop[i] < 40):
                signals[i] = 0.25
                position = 1
            # Enter short: KAMA down, RSI < 50, Chop < 40 (trending down)
            elif (kama_1d_aligned[i] < kama_1d_aligned[i-1] and 
                  rsi[i] < 50 and 
                  chop[i] < 40):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: RSI < 50 or Chop > 50 (range or reversal)
            if (rsi[i] < 50) or (chop[i] > 50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: RSI > 50 or Chop > 50 (range or reversal)
            if (rsi[i] > 50) or (chop[i] > 50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals