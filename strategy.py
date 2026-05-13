#!/usr/bin/env python3
# Hypothesis: 1d KAMA trend with RSI mean reversion and chop regime filter.
# Long when KAMA rising (bullish trend), RSI < 30 (oversold), and choppy market (CHOP > 61.8).
# Short when KAMA falling (bearish trend), RSI > 70 (overbought), and choppy market (CHOP > 61.8).
# Exit on opposite RSI extreme (RSI > 70 for longs, RSI < 30 for shorts) or chop regime ends (CHOP < 38.2).
# Uses 1d primary timeframe and 1w HTF for trend alignment via EMA34.
# Designed for BTC/ETH with mean reversion in choppy markets, targeting 15-25 trades/year.

name = "1d_KAMA_RSI_Chop_Regime_v1"
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
    
    # Calculate ER (Efficiency Ratio) for KAMA over 10 periods
    change = np.abs(np.diff(close, n=10))  # |close[t] - close[t-10]|
    change = np.concatenate([[np.nan] * 10, change])  # align length
    volatility = np.sum(np.abs(np.diff(close)), axis=0)  # sum of |close[t] - close[t-1]| over 10 periods
    volatility = pd.Series(volatility).rolling(window=10, min_periods=1).sum().values
    er = np.where(volatility > 0, change / volatility, 0)
    
    # Calculate Smoothing Constants (SC) for KAMA
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.full(n, np.nan)
    kama[9] = close[9]  # seed at index 9 (after 10 periods)
    for i in range(10, n):
        if np.isnan(kama[i-1]) or np.isnan(sc[i]):
            kama[i] = close[i]
        else:
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Calculate RSI(14)
    delta = np.diff(close)
    delta = np.concatenate([[np.nan], delta])  # align length
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss > 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate Choppiness Index (CHOP) over 14 periods
    # True Range (TR)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first bar
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    # Max/Min close over 14 periods
    max_close = pd.Series(close).rolling(window=14, min_periods=14).max().values
    min_close = pd.Series(close).rolling(window=14, min_periods=14).min().values
    chop = np.where(atr_sum > 0, 100 * np.log10(atr_sum / (max_close - min_close)) / np.log10(14), 50)
    chop = np.where((max_close - min_close) == 0, 50, chop)  # avoid division by zero
    
    # Get 1w data for EMA34 trend filter (MTF)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate EMA34 on 1w close
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align HTF arrays to 1d timeframe (wait for completed 1w bar)
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after sufficient data for indicators
        # Skip if any required data is NaN
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or 
            np.isnan(ema34_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: KAMA rising (bullish trend), RSI < 30 (oversold), choppy market (CHOP > 61.8)
            if kama[i] > kama[i-1] and rsi[i] < 30 and chop[i] > 61.8:
                signals[i] = 0.25
                position = 1
            # SHORT: KAMA falling (bearish trend), RSI > 70 (overbought), choppy market (CHOP > 61.8)
            elif kama[i] < kama[i-1] and rsi[i] > 70 and chop[i] > 61.8:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: RSI > 70 (overbought) OR chop regime ends (CHOP < 38.2)
            if rsi[i] > 70 or chop[i] < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: RSI < 30 (oversold) OR chop regime ends (CHOP < 38.2)
            if rsi[i] < 30 or chop[i] < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals