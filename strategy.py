#!/usr/bin/env python3
"""
4h_1d_KAMA_Trend_With_RSI_Pullback
Hypothesis: Combines KAMA trend direction with RSI pullback entries on 4h, filtered by 1d trend alignment.
Long when KAMA rising, RSI < 30, and price above 1d EMA200; short when KAMA falling, RSI > 70, and price below 1d EMA200.
Designed for low trade frequency by requiring trend alignment and oversold/overbought conditions.
Works in bull via buying dips in uptrend, in bear via selling rallies in downtrend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_KAMA_Trend_With_RSI_Pullback"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # === DAILY DATA ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # EMA200 for 1d trend filter
    close_1d_s = pd.Series(close_1d)
    ema200_1d = close_1d_s.ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # === 4H INDICATORS ===
    # KAMA components
    close_s = pd.Series(close)
    change = abs(close_s - close_s.shift(10))
    volatility = abs(close_s.diff()).rolling(window=10, min_periods=10).sum()
    er = change / volatility.replace(0, np.nan)
    er = er.fillna(0).values
    sc = (er * (0.6645 - 0.0645) + 0.0645) ** 2
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # KAMA direction
    kama_rising = kama > np.roll(kama, 1)
    kama_falling = kama < np.roll(kama, 1)
    kama_rising[0] = False
    kama_falling[0] = False
    
    # RSI(14)
    delta = close_s.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(14, n):
        # Skip if not ready
        if np.isnan(ema200_1d_aligned[i]) or np.isnan(kama[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Trend filter: price vs 1d EMA200
        price_above_ema200 = close[i] > ema200_1d_aligned[i]
        price_below_ema200 = close[i] < ema200_1d_aligned[i]
        
        # Entry conditions
        long_setup = kama_rising[i] and rsi[i] < 30 and price_above_ema200
        short_setup = kama_falling[i] and rsi[i] > 70 and price_below_ema200
        
        # Exit when trend fails or RSI normalizes
        exit_long = not price_above_ema200 or rsi[i] > 50
        exit_short = not price_below_ema200 or rsi[i] < 50
        
        if long_setup and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_setup and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals