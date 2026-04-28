#!/usr/bin/env python3
"""
12h_KAMA_Direction_RSI_Chop_Filter
Hypothesis: Use Kaufman Adaptive Moving Average (KAMA) for trend direction on 12h timeframe, filtered by RSI momentum and Choppiness Index regime filter to avoid whipsaws in sideways markets. Designed for 12h timeframe to achieve 12-37 trades/year with strong trend capture in both bull and bear markets while minimizing false signals.
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
    
    # Get 1d data for Choppiness Index calculation
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 14-period Choppiness Index on daily data
    # CHOP = 100 * log10(sum(ATR(14)) / (max(high,14) - min(low,14))) / log10(14)
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr14 = tr.rolling(window=14, min_periods=14).mean()
    
    sum_atr14 = atr14.rolling(window=14, min_periods=14).sum()
    max_high14 = df_1d['high'].rolling(window=14, min_periods=14).max()
    min_low14 = df_1d['low'].rolling(window=14, min_periods=14).min()
    range14 = max_high14 - min_low14
    
    # Avoid division by zero
    chop_raw = 100 * (np.log10(sum_atr14) - np.log10(range14)) / np.log10(14)
    chop = chop_raw.fillna(50).values  # Fill NaN with neutral value
    
    # Get 12h data for KAMA and RSI
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate Kaufman Adaptive Moving Average (KAMA) with ER=10
    # Efficiency Ratio = |close - close[10]| / sum(|close - close[-1]| for 10 periods)
    close_12h = df_12h['close']
    change = abs(close_12h - close_12h.shift(10))
    volatility = abs(close_12h - close_12h.shift(1)).rolling(window=10, min_periods=10).sum()
    er = change / volatility.replace(0, np.nan)
    er = er.fillna(0).values
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)  # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.full_like(close_12h.values, np.nan)
    kama[0] = close_12h.iloc[0]
    for i in range(1, len(kama)):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close_12h.iloc[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # Calculate 14-period RSI on 12h data
    delta = close_12h.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values  # Fill NaN with neutral value
    
    # Align all higher timeframe data to 12h (which is our primary timeframe)
    kama_aligned = align_htf_to_ltf(prices, df_12h, kama)
    rsi_aligned = align_htf_to_ltf(prices, df_12h, rsi)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend direction from KAMA
        kama_bullish = close[i] > kama_aligned[i]
        kama_bearish = close[i] < kama_aligned[i]
        
        # RSI conditions: avoid extreme overbought/oversold
        rsi_not_overbought = rsi_aligned[i] < 70
        rsi_not_oversold = rsi_aligned[i] > 30
        
        # Choppiness Index filter: only trade in trending markets (CHOP < 38.2) or strong mean reversion (CHOP > 61.8)
        chop_trending = chop_aligned[i] < 38.2
        chop_ranging = chop_aligned[i] > 61.8
        
        # Entry conditions
        # Long: KAMA bullish + RSI not overbought + (trending OR strong ranging for mean reversion)
        long_entry = (kama_bullish and 
                     rsi_not_overbought and 
                     (chop_trending or chop_ranging))
        
        # Short: KAMA bearish + RSI not oversold + (trending OR strong ranging for mean reversion)
        short_entry = (kama_bearish and 
                      rsi_not_oversold and 
                      (chop_trending or chop_ranging))
        
        # Exit conditions: reverse when opposite signal occurs
        long_exit = kama_bearish  # Exit long when KAMA turns bearish
        short_exit = kama_bullish  # Exit short when KAMA turns bullish
        
        if long_entry and position <= 0:
            signals[i] = 0.30
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.30
            position = -1
        elif long_exit and position == 1:
            signals[i] = -0.30  # Reverse to short
            position = -1
        elif short_exit and position == -1:
            signals[i] = 0.30   # Reverse to long
            position = 1
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.30
            elif position == -1:
                signals[i] = -0.30
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_KAMA_Direction_RSI_Chop_Filter"
timeframe = "12h"
leverage = 1.0