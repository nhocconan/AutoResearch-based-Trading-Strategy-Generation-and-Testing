#!/usr/bin/env python3
"""
1d_KAMA_Trend_RSI_ChopFilter_v1
Hypothesis: 1d KAMA trend direction + RSI(14) extreme + choppiness regime filter (CHOP > 61.8 = range) for mean reversion entries. Designed for low trade frequency (~10-25/year) to minimize fee drag and work in both bull (trend continuation) and bear (mean reversion in range) markets. Uses 1d primary timeframe with 1w HTF for trend context.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # === 1w trend filter: 34-period EMA ===
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # === 1d indicators: KAMA, RSI, Chop ===
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # KAMA (adaptive trend)
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.abs(np.diff(close))
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (2/2 - 2/30) + 2/30) ** 2  # fast=2, slow=30
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(span=14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(span=14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Choppiness Index (14)
    atr = np.zeros_like(close)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    sum_atr = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    chop = 100 * np.log10(sum_atr / (highest_high - lowest_low)) / np.log10(14)
    chop = np.where((highest_high - lowest_low) != 0, chop, 50)  # avoid division by zero
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(ema_34_1w_aligned[i]) or
            np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = close[i]
        trend_1w = ema_34_1w_aligned[i]
        kama_val = kama[i]
        rsi_val = rsi[i]
        chop_val = chop[i]
        
        if position == 0:
            # Mean reversion long in range: price below KAMA + RSI oversold + choppy market
            if price_close < kama_val and rsi_val < 30 and chop_val > 61.8:
                signals[i] = 0.25
                position = 1
            # Mean reversion short in range: price above KAMA + RSI overbought + choppy market
            elif price_close > kama_val and rsi_val > 70 and chop_val > 61.8:
                signals[i] = -0.25
                position = -1
            # Trend continuation: strong trend + price pulls back to KAMA
            elif price_close > kama_val and price_close > trend_1w and rsi_val > 50 and rsi_val < 70:
                signals[i] = 0.25
                position = 1
            elif price_close < kama_val and price_close < trend_1w and rsi_val < 50 and rsi_val > 30:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: reverse signal or RSI extreme reversal
            if position == 1 and (price_close > kama_val * 1.02 or rsi_val > 70):
                signals[i] = 0.0
                position = 0
            elif position == -1 and (price_close < kama_val * 0.98 or rsi_val < 30):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_KAMA_Trend_RSI_ChopFilter_v1"
timeframe = "1d"
leverage = 1.0