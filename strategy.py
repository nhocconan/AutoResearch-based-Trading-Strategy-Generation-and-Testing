#!/usr/bin/env python3
# 1d_KAMA_Trend_RSI_Chop
# Hypothesis: KAMA adapts to market noise, capturing strong trends while avoiding whipsaws in ranges.
# Long when KAMA rises + RSI > 50 + Choppiness < 61.8 (trending regime).
# Short when KAMA falls + RSI < 50 + Choppiness < 61.8.
# Uses 1w trend filter to avoid counter-trend trades in major reversals.
# Designed for low-frequency, high-conviction trades on daily timeframe.

name = "1d_KAMA_Trend_RSI_Chop"
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

    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # KAMA calculation (ER=10, fast=2, slow=30)
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.abs(np.diff(close)).cumsum()
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Align KAMA to lower timeframe (no delay needed as it's synchronous)
    kama_aligned = align_htf_to_ltf(prices, df_1w, kama)
    
    # RSI (14) on daily closes
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Choppiness Index (14)
    atr = np.zeros_like(close)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = high[0] - low[0]
    tr2[0] = np.abs(high[0] - close[0])
    tr3[0] = np.abs(low[0] - close[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    sum_atr = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    chop = np.where((highest_high - lowest_low) != 0, 
                    100 * np.log10(sum_atr / (highest_high - lowest_low)) / np.log10(14), 
                    50)
    
    # 1-week EMA34 for trend filter
    ema34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(100, n):
        # Skip if any required value is NaN
        if (np.isnan(kama_aligned[i]) or 
            np.isnan(rsi[i]) or 
            np.isnan(chop[i]) or 
            np.isnan(ema34_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Trend conditions
        kama_rising = kama_aligned[i] > kama_aligned[i-1]
        kama_falling = kama_aligned[i] < kama_aligned[i-1]
        
        # Momentum filter
        rsi_bullish = rsi[i] > 50
        rsi_bearish = rsi[i] < 50
        
        # Regime filter: trending market (Choppiness < 61.8)
        trending = chop[i] < 61.8
        
        # Weekly trend filter
        weekly_uptrend = close[i] > ema34_1w_aligned[i]
        weekly_downtrend = close[i] < ema34_1w_aligned[i]

        if position == 0:
            # LONG: KAMA rising + RSI > 50 + trending regime + weekly uptrend
            if kama_rising and rsi_bullish and trending and weekly_uptrend:
                signals[i] = 0.25
                position = 1
            # SHORT: KAMA falling + RSI < 50 + trending regime + weekly downtrend
            elif kama_falling and rsi_bearish and trending and weekly_downtrend:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: KAMA falling OR RSI < 50 OR ranging regime
            if not kama_rising or not rsi_bullish or not trending:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: KAMA rising OR RSI > 50 OR ranging regime
            if not kama_falling or not rsi_bearish or not trending:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals