#!/usr/bin/env python3
"""
4h_1d_KAMA_RSI_Chop_v1
Hypothesis: 4h timeframe with KAMA direction (trend), RSI for momentum, and Choppiness Index for regime filter.
Works in bull/bear markets by only taking trades when trend is aligned (KAMA) and momentum is confirmed (RSI),
while avoiding choppy markets (Chop > 61.8) where whipsaws occur. Designed for ~25-40 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_KAMA_RSI_Chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Load 1d data ONCE before loop for KAMA, RSI, and Choppiness Index
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate KAMA ( Kaufman Adaptive Moving Average ) on 1d
    # ER = Efficiency Ratio, SC = Smoothing Constant
    close_1d = df_1d['close'].values
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.sum(np.abs(np.diff(close_1d, prepend=close_1d[0])), axis=0)  # placeholder, will fix below
    
    # Proper ER calculation
    price_change = np.abs(np.diff(close_1d, 10))  # 10-period change
    volatility_sum = np.zeros_like(close_1d)
    for i in range(len(close_1d)):
        if i < 10:
            volatility_sum[i] = np.nan
        else:
            volatility_sum[i] = np.sum(np.abs(np.diff(close_1d[i-10:i+1])))
    er = np.where(volatility_sum != 0, price_change / volatility_sum, 0)
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    kama = np.full_like(close_1d, np.nan)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        if np.isnan(sc[i]) or np.isnan(kama[i-1]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # Calculate RSI(14) on 1d
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate Choppiness Index(14) on 1d
    atr = np.zeros_like(close_1d)
    tr1 = np.abs(np.diff(df_1d['high'].values))
    tr2 = np.abs(np.diff(df_1d['low'].values))
    tr3 = np.abs(np.diff(df_1d['close'].values))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    max_high = pd.Series(df_1d['high'].values).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(df_1d['low'].values).rolling(window=14, min_periods=14).min().values
    chop = np.where((max_high - min_low) != 0, 
                    100 * np.log10(atr.sum() / (max_high - min_low)) / np.log10(14), 
                    50)
    # Fix chop calculation - proper rolling sum
    chop = np.full_like(close_1d, np.nan)
    for i in range(14, len(close_1d)):
        atr_sum = np.sum(tr[i-13:i+1])
        range_hl = max_high[i] - min_low[i]
        if range_hl != 0:
            chop[i] = 100 * np.log10(atr_sum / range_hl) / np.log10(14)
        else:
            chop[i] = 50
    
    # Align indicators to 4h timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # 4h EMA(20) for entry timing
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(chop_aligned[i]) or np.isnan(ema_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Regime filter: avoid choppy markets (Chop > 61.8)
        in_trend = chop_aligned[i] <= 61.8
        
        # Trend direction: price relative to KAMA
        above_kama = close[i] > kama_aligned[i]
        below_kama = close[i] < kama_aligned[i]
        
        # Momentum: RSI not overbought/oversold
        rsi_not_extreme = (rsi_aligned[i] > 30) and (rsi_aligned[i] < 70)
        
        # Entry conditions: trend-aligned pullback to EMA with momentum
        long_entry = (close[i] > ema_20[i]) and above_kama and rsi_not_extreme and in_trend
        short_entry = (close[i] < ema_20[i]) and below_kama and rsi_not_extreme and in_trend
        
        # Exit conditions: opposite KAMA cross or Chop > 61.8
        long_exit = (close[i] < kama_aligned[i]) or (chop_aligned[i] > 61.8)
        short_exit = (close[i] > kama_aligned[i]) or (chop_aligned[i] > 61.8)
        
        # Priority: entry > exit > hold
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals