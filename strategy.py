#!/usr/bin/env python3
# 1d_1w_kama_rsi_chop_v1
# Strategy: 1d KAMA trend direction with RSI and Choppiness regime filter
# Timeframe: 1d
# Leverage: 1.0
# Hypothesis: KAMA adapts to market noise, providing reliable trend signals.
# RSI filters for momentum strength (>50 bullish, <50 bearish).
# Choppiness index identifies ranging markets (CHOP > 61.8) where we avoid trend signals.
# Works in bull markets via trend following and bear markets via inverse signals.
# Low trade frequency expected due to multiple confirmation requirements.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_kama_rsi_chop_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # KAMA(10) for trend direction
    # Efficiency Ratio
    change = np.abs(np.diff(close, n=10))
    volatility = np.sum(np.abs(np.diff(close)), axis=1)
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1))**2
    # KAMA calculation
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI(14) for momentum
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Choppiness Index(14) for regime
    atr = np.zeros(n)
    tr1 = np.abs(high - low)
    tr2 = np.abs(np.roll(high, 1) - close)
    tr3 = np.abs(np.roll(low, 1) - close)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    chop = np.zeros(n)
    for i in range(14, n):
        if highest_high[i] != lowest_low[i]:
            sum_atr = np.sum(atr[i-13:i+1])  # 14-period sum
            chop[i] = 100 * np.log10(sum_atr / (highest_high[i] - lowest_low[i])) / np.log10(14)
        else:
            chop[i] = 50  # neutral when no range
    
    # Align 1w indicators to 1d timeframe
    kama_1w = pd.Series(df_1w['close']).ewm(alpha=2/(2+1), adjust=False).mean().values  # KAMA(2,30) on weekly
    kama_1w_aligned = align_htf_to_ltf(prices, df_1w, kama_1w)
    
    chop_1w = np.zeros(len(df_1w))
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly choppiness
    atr_1w = np.zeros(len(df_1w))
    tr1_1w = np.abs(high_1w - low_1w)
    tr2_1w = np.abs(np.roll(high_1w, 1) - close_1w)
    tr3_1w = np.abs(np.roll(low_1w, 1) - close_1w)
    tr_1w = np.maximum(tr1_1w, np.maximum(tr2_1w, tr3_1w))
    atr_1w = pd.Series(tr_1w).rolling(window=14, min_periods=14).mean().values
    
    highest_high_1w = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
    lowest_low_1w = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
    
    for i in range(14, len(df_1w)):
        if highest_high_1w[i] != lowest_low_1w[i]:
            sum_atr_1w = np.sum(atr_1w[i-13:i+1])
            chop_1w[i] = 100 * np.log10(sum_atr_1w / (highest_high_1w[i] - lowest_low_1w[i])) / np.log10(14)
        else:
            chop_1w[i] = 50
    
    chop_1w_aligned = align_htf_to_ltf(prices, df_1w, chop_1w, additional_delay_bars=0)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or \
           np.isnan(kama_1w_aligned[i]) or np.isnan(chop_1w_aligned[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Regime filter: avoid trend signals in choppy markets (both timeframes)
        regime_filter = (chop[i] <= 61.8) and (chop_1w_aligned[i] <= 61.8)
        
        # KAMA trend direction
        kama_bullish = kama[i] > close[i]
        kama_bearish = kama[i] < close[i]
        
        # Weekly KAMA filter for higher timeframe alignment
        weekly_bullish = kama_1w_aligned[i] > df_1w['close'].iloc[-1] if len(df_1w) > 0 else False  # simplified
        weekly_bullish = kama_1w_aligned[i] > close[i]  # compare to current price
        
        # Entry conditions
        # Long: KAMA bullish AND RSI bullish AND favorable regime
        if kama_bullish and rsi[i] > 50 and regime_filter and position != 1:
            position = 1
            signals[i] = 0.25
        # Short: KAMA bearish AND RSI bearish AND favorable regime
        elif kama_bearish and rsi[i] < 50 and regime_filter and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: Opposite KAMA signal or regime becomes too choppy
        elif position == 1 and (not kama_bullish or chop[i] > 61.8):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (not kama_bearish or chop[i] > 61.8):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals