#!/usr/bin/env python3
"""
1d_KAMA_RSI_Chop_Filter_v1
Hypothesis: KAMA trend direction + RSI extremes + Choppiness regime filter captures trending moves while avoiding whipsaws.
Designed for very low trade frequency (target 10-25/year) to minimize fee drag and work in both bull and bear markets.
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
    
    # KAMA (ER=10) - adaptive trend
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.abs(np.diff(close))
    er = np.divide(change, volatility, out=np.zeros_like(change), where=volatility!=0)
    sc = (er * (2/2 - 2/30) + 2/30) ** 2  # fast=2, slow=30
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Choppiness Index(14) - regime filter
    atr1 = np.abs(high - low)
    atr2 = np.abs(high - np.roll(close, 1))
    atr3 = np.abs(low - np.roll(close, 1))
    atr = np.maximum(np.maximum(atr1, atr2), atr3)
    atr[0] = atr1[0]
    tr_sum = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    highest = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(tr_sum / (highest - lowest)) / np.log10(14)
    
    # 1w trend filter: EMA34
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need KAMA (30), RSI (14), Chop (14), EMA34 (34)
    start_idx = max(30, 14, 14, 34)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or 
            np.isnan(ema34_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        kama_val = kama[i]
        rsi_val = rsi[i]
        chop_val = chop[i]
        ema34 = ema34_1w_aligned[i]
        
        if position == 0:
            # Trend filter: price vs weekly EMA34
            uptrend = close_val > ema34
            downtrend = close_val < ema34
            
            # Regime filter: only trade in trending markets (Chop < 38.2) or extreme mean reversion (Chop > 61.8)
            trending = chop_val < 38.2
            ranging = chop_val > 61.8
            
            if uptrend and trending:
                # Long: KAMA bullish + RSI not overbought
                if close_val > kama_val and rsi_val < 70:
                    signals[i] = size
                    position = 1
            elif downtrend and trending:
                # Short: KAMA bearish + RSI not oversold
                if close_val < kama_val and rsi_val > 30:
                    signals[i] = -size
                    position = -1
            elif ranging:
                # Mean reversion in ranging market
                if rsi_val < 30 and close_val > kama_val:
                    signals[i] = size
                    position = 1
                elif rsi_val > 70 and close_val < kama_val:
                    signals[i] = -size
                    position = -1
        elif position == 1:
            # Exit: RSI overbought or trend change
            if rsi_val > 70 or close_val < kama_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit: RSI oversold or trend change
            if rsi_val < 30 or close_val > kama_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_KAMA_RSI_Chop_Filter_v1"
timeframe = "1d"
leverage = 1.0