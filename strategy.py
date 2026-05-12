#!/usr/bin/env python3
"""
1d_KAMA_With_RSI_And_Chop_Regime
Adapted from experiment #159624 which had 0 trades due to tight conditions.
Now uses 1d timeframe with KAMA trend, RSI momentum, and Choppiness index regime filter.
KAMA adapts to market noise - effective in both trending and ranging markets.
RSI provides momentum confirmation. Choppiness index filters for trending regimes (CHOP < 38.2).
Designed for low trade frequency: target 30-100 total trades over 4 years.
Works in bull/bear markets by following KAMA trend direction only in trending regimes.
"""

name = "1d_KAMA_With_RSI_And_Chop_Regime"
timeframe = "1d"
leverage = 1.0

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
    
    # KAMA (Kaufman Adaptive Moving Average) - adapts to market noise
    # ER = Efficiency Ratio = |change| / volatility
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close, prepend=close[0])), axis=0)  # placeholder, will fix below
    
    # Proper KAMA calculation
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.zeros_like(change)
    for i in range(1, len(volatility)):
        volatility[i] = volatility[i-1] + np.abs(close[i] - close[i-1]) - np.abs(close[i-10] - close[i-10]) if i >= 10 else volatility[i-1] + np.abs(close[i] - close[i-1])
    # Simplified: use rolling sum of absolute changes
    volatility = pd.Series(close).diff().abs().rolling(window=10, min_periods=10).sum().values
    volatility = np.where(volatility == 0, 1, volatility)  # avoid division by zero
    er = change / volatility
    er = np.where(np.isnan(er), 0, er)
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # for EMA(2)
    slow_sc = 2 / (30 + 1)  # for EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    sc = np.where(np.isnan(sc), slow_sc, sc)
    
    # Calculate KAMA
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI (14-period)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss == 0, 100, avg_gain / avg_loss)
    rsi = 100 - (100 / (1 + rs))
    rsi = np.where(np.isnan(rsi), 50, rsi)  # neutral when undefined
    
    # Choppiness Index (14-period) - identifies trending vs ranging markets
    # CHOP > 61.8 = ranging, CHOP < 38.2 = trending
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]  # first TR
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = np.where((max_high - min_low) == 0, 50, 100 * np.log10(atr14.sum() / (max_high - min_low)) / np.log10(14))
    # Fix the chop calculation - proper rolling sum
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    range_14 = max_high - min_low
    chop = np.where(range_14 == 0, 50, 100 * np.log10(atr_sum / range_14) / np.log10(14))
    chop = np.where(np.isnan(chop), 50, chop)
    
    # Weekly trend filter for higher timeframe context
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Align KAMA, RSI, Chop to daily timeframe (already aligned as we calculated on close)
    # But we need to ensure no look-ahead - KAMA, RSI, CHOP use only past data
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # start after warmup period
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or 
            np.isnan(ema_50_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Only trade in trending regimes (Choppiness < 38.2)
        is_trending = chop[i] < 38.2
        
        if position == 0:
            # LONG: Price above KAMA + RSI > 50 (bullish momentum) + weekly uptrend + trending regime
            if (close[i] > kama[i] and 
                rsi[i] > 50 and 
                close[i] > ema_50_1w_aligned[i] and 
                is_trending):
                signals[i] = 0.25
                position = 1
            # SHORT: Price below KAMA + RSI < 50 (bearish momentum) + weekly downtrend + trending regime
            elif (close[i] < kama[i] and 
                  rsi[i] < 50 and 
                  close[i] < ema_50_1w_aligned[i] and 
                  is_trending):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price below KAMA OR RSI < 40 (losing momentum) OR chop > 50 (ranging)
            if (close[i] < kama[i]) or (rsi[i] < 40) or (chop[i] > 50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price above KAMA OR RSI > 60 (losing momentum) OR chop > 50 (ranging)
            if (close[i] > kama[i]) or (rsi[i] > 60) or (chop[i] > 50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals