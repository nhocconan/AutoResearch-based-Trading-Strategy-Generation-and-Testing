#!/usr/bin/env python3
"""
1d_KAMA_Direction_RSI_Chop_Filter_v1
Hypothesis: KAMA direction (trend) on 1d, filtered by RSI momentum and Choppiness index (range) on 1d.
In uptrend (price > KAMA), go long when RSI > 50 and CHOP > 61.8 (range) for mean reversion longs.
In downtrend (price < KAMA), go short when RSI < 50 and CHOP > 61.8 (range) for mean reversion shorts.
Uses weekly trend filter: only take longs when price > weekly EMA34, shorts when price < weekly EMA34.
Designed for low trade frequency and high edge in ranging markets with weekly trend bias.
"""

name = "1d_KAMA_Direction_RSI_Chop_Filter_v1"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get 1d data for KAMA, RSI, CHOP
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # KAMA calculation (ER=10, fast=2, slow=30)
    close_1d = df_1d['close'].values
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.sum(np.abs(np.diff(close_1d, 1)), axis=0)  # needs correction
    # Correct volatility calculation: sum of absolute changes over 10 periods
    volatility = pd.Series(close_1d).rolling(window=10, min_periods=10).apply(
        lambda x: np.sum(np.abs(np.diff(x))), raw=True
    ).values
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama = np.full_like(close_1d, np.nan)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # RSI(14) on 1d
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Choppiness Index (14)
    atr_1d = pd.Series(high - low).rolling(window=14, min_periods=14).mean().values
    sum_atr = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(sum_atr / (max_high - min_low)) / np.log10(14)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    ema_34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Align 1d indicators to 1d timeframe (no alignment needed as same timeframe)
    kama_aligned = kama
    rsi_aligned = rsi
    chop_aligned = chop
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # start after warmup
        # Skip if any key value is NaN
        if np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or np.isnan(chop_aligned[i]) or np.isnan(ema_34_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Weekly trend filter
        weekly_uptrend = close[i] > ema_34_1w_aligned[i]
        weekly_downtrend = close[i] < ema_34_1w_aligned[i]
        
        # KAMA direction
        kama_uptrend = close[i] > kama_aligned[i]
        kama_downtrend = close[i] < kama_aligned[i]
        
        # RSI momentum
        rsi_bullish = rsi_aligned[i] > 50
        rsi_bearish = rsi_aligned[i] < 50
        
        # Choppiness filter (range market)
        chop_range = chop_aligned[i] > 61.8
        
        if position == 0:
            # LONG: weekly uptrend + KAMA uptrend + RSI bullish + chop range
            if weekly_uptrend and kama_uptrend and rsi_bullish and chop_range:
                signals[i] = 0.25
                position = 1
            # SHORT: weekly downtrend + KAMA downtrend + RSI bearish + chop range
            elif weekly_downtrend and kama_downtrend and rsi_bearish and chop_range:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: weekly trend turns down OR KAMA turns down OR RSI < 40
            if not (weekly_uptrend and kama_uptrend and rsi_aligned[i] > 40):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: weekly trend turns up OR KAMA turns up OR RSI > 60
            if not (weekly_downtrend and kama_downtrend and rsi_aligned[i] < 60):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals