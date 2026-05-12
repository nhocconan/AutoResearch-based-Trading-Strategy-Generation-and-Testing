#!/usr/bin/env python3
# 4h_KAMA_Trend_RSI_Chop_Filter
# Hypothesis: KAMA (Kaufman Adaptive Moving Average) adapts to market noise, providing a dynamic trend filter.
# In trending markets, KAMA closely follows price; in ranging markets, it smooths out noise.
# Combined with RSI for momentum and Choppiness Index for regime detection, this strategy aims to capture
# strong trends while avoiding whipsaws in choppy conditions. Works in both bull and bear markets by
# following the adaptive trend and using RSI extremes for entry timing.

name = "4h_KAMA_Trend_RSI_Chop_Filter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === KAMA (10, 2, 30) ===
    close_s = pd.Series(close)
    change = abs(close_s - close_s.shift(10))
    volatility = abs(close_s - close_s.shift(1)).rolling(window=10, min_periods=10).sum()
    er = change / volatility.replace(0, np.nan)
    sc = (er * (2/2 - 2/30) + 2/30) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        if np.isnan(sc[i]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # === RSI (14) ===
    delta = pd.Series(close).diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    
    # === Choppiness Index (14) ===
    atr = pd.Series(np.maximum(high - low, np.maximum(abs(high - close_s.shift(1)), abs(low - close_s.shift(1))))).rolling(window=14, min_periods=14).mean()
    high_roll = pd.Series(high).rolling(window=14, min_periods=14).max()
    low_roll = pd.Series(low).rolling(window=14, min_periods=14).min()
    chop = 100 * np.log10((atr.rolling(window=14, min_periods=14).sum() / (high_roll - low_roll))) / np.log10(14)
    
    # === 1d Trend Filter (EMA 34) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or np.isnan(ema_34_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 1d EMA34
        price_above_ema = close[i] > ema_34_1d_aligned[i]
        price_below_ema = close[i] < ema_34_1d_aligned[i]
        
        # KAMA trend: price above/below KAMA
        price_above_kama = close[i] > kama[i]
        price_below_kama = close[i] < kama[i]
        
        # RSI conditions
        rsi_oversold = rsi[i] < 30
        rsi_overbought = rsi[i] > 70
        
        # Chop regime: chop > 61.8 = ranging, chop < 38.2 = trending
        chop_trending = chop[i] < 38.2
        chop_ranging = chop[i] > 61.8
        
        if position == 0:
            # LONG: price above both KAMA and 1d EMA, RSI oversold, trending market
            if price_above_kama and price_above_ema and rsi_oversold and chop_trending:
                signals[i] = 0.25
                position = 1
            # SHORT: price below both KAMA and 1d EMA, RSI overbought, trending market
            elif price_below_kama and price_below_ema and rsi_overbought and chop_trending:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: price below KAMA or RSI overbought
            if price_below_kama or rsi[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price above KAMA or RSI oversold
            if price_above_kama or rsi[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals