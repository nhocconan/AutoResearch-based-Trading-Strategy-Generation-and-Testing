#!/usr/bin/env python3
name = "1d_KAMA_Trend_RSI_Chop_Filter"
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
    
    # Weekly trend: EMA50
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    weekly_uptrend = close > ema_1w_aligned
    
    # Daily KAMA
    price_series = pd.Series(close)
    delta = price_series.diff().abs()
    direction = abs(price_series.diff(10))
    volatility = delta.rolling(window=10, min_periods=10).sum()
    er = direction / volatility.replace(0, 1e-10)
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1))**2
    kama = [0] * len(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    kama = np.array(kama)
    kama_aligned = kama  # KAMA is already on daily timeframe
    
    # Daily RSI(14)
    delta_rsi = pd.Series(close).diff()
    gain = delta_rsi.where(delta_rsi > 0, 0)
    loss = -delta_rsi.where(delta_rsi < 0, 0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    
    # Choppiness Index (14)
    atr1 = np.maximum(high - low, np.maximum(abs(high - np.roll(close, 1)), abs(low - np.roll(close, 1))))
    atr1[0] = high[0] - low[0]
    atr_sum = pd.Series(atr1).rolling(window=14, min_periods=14).sum().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(14)
    chop = np.nan_to_num(chop, nan=50.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for weekly EMA and calculations
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(ema_1w_aligned[i]) or np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Price > KAMA + Weekly uptrend + RSI > 50 + Chop < 61.8 (trending)
            if close[i] > kama[i] and weekly_uptrend[i] and rsi[i] > 50 and chop[i] < 61.8:
                signals[i] = 0.25
                position = 1
            # Short: Price < KAMA + Weekly downtrend + RSI < 50 + Chop < 61.8 (trending)
            elif close[i] < kama[i] and not weekly_uptrend[i] and rsi[i] < 50 and chop[i] < 61.8:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Price < KAMA or RSI < 40 or Chop > 61.8 (ranging)
            if close[i] < kama[i] or rsi[i] < 40 or chop[i] > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Price > KAMA or RSI > 60 or Chop > 61.8 (ranging)
            if close[i] > kama[i] or rsi[i] > 60 or chop[i] > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals