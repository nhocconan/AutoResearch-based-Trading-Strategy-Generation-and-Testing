#!/usr/bin/env python3
"""
1d_KAMA_Direction_RSI_ChopFilter
1d strategy using KAMA direction, RSI momentum, and Chop filter.
- Long: KAMA trending up + RSI > 50 + Chop < 61.8 (trending regime)
- Short: KAMA trending down + RSI < 50 + Chop < 61.8 (trending regime)
- Exit: Opposite signal or Chop > 61.8 (range regime)
Designed for ~10-20 trades/year per symbol (40-80 total over 4 years)
Works in bull trends (KAMA up + RSI > 50) and bear trends (KAMA down + RSI < 50)
Avoids choppy markets via Chop filter
"""

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
    
    # Get 1d data for KAMA, RSI, and Chop
    df_1d = get_htf_data(prices, '1d')
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # KAMA on 1d
    # Efficiency Ratio
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.abs(np.diff(close_1d))
    er = np.divide(change, volatility, out=np.zeros_like(change), where=volatility!=0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1))**2
    # KAMA calculation
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # RSI on 1d
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Chop on 1d
    atr = np.zeros_like(close_1d)
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    sum_atr14 = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    max_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.divide(np.log10(sum_atr14) / np.log10(2), 
                           np.log10((max_high - min_low) / atr), 
                           out=np.zeros_like(sum_atr14), 
                           where=((max_high - min_low) > 0) & (atr > 0))
    
    # Align to original timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # need enough for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # KAMA direction
        kama_up = kama_aligned[i] > kama_aligned[i-1]
        kama_down = kama_aligned[i] < kama_aligned[i-1]
        
        # RSI condition
        rsi_above_50 = rsi_aligned[i] > 50
        rsi_below_50 = rsi_aligned[i] < 50
        
        # Chop regime (trending when < 61.8)
        trending = chop_aligned[i] < 61.8
        ranging = chop_aligned[i] >= 61.8
        
        if position == 0:
            # Long: KAMA up + RSI > 50 + trending
            if kama_up and rsi_above_50 and trending:
                signals[i] = 0.25
                position = 1
            # Short: KAMA down + RSI < 50 + trending
            elif kama_down and rsi_below_50 and trending:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: KAMA down OR RSI < 50 OR ranging
            if not (kama_up and rsi_above_50 and trending):
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: KAMA up OR RSI > 50 OR ranging
            if not (kama_down and rsi_below_50 and trending):
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_KAMA_Direction_RSI_ChopFilter"
timeframe = "1d"
leverage = 1.0