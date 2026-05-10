#!/usr/bin/env python3
# 1D_KAMA_RSI_Chop_Filter
# Hypothesis: Use KAMA to determine daily trend direction, RSI for mean reversion, and Choppiness Index to filter ranging vs trending markets.
# Long when: KAMA trending up, RSI < 40, and Choppiness > 61.8 (ranging market).
# Short when: KAMA trending down, RSI > 60, and Choppiness > 61.8 (ranging market).
# Uses weekly trend filter: only trade in direction of weekly EMA200 trend.
# Works in bull/bear by following weekly trend and using RSI extremes in ranging markets.
# Target: 10-25 trades/year per symbol.

name = "1D_KAMA_RSI_Chop_Filter"
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
    
    # KAMA for trend direction (ER=10, fast=2, slow=30)
    close_s = pd.Series(close)
    change = abs(close_s.diff(10))
    volatility = close_s.diff().abs().rolling(10).sum()
    er = change / volatility.replace(0, np.nan)
    sc = (er * (2/2 - 2/30) + 2/30) ** 2
    kama = [close[0]]
    for i in range(1, len(close)):
        kama.append(kama[-1] + sc.iloc[i] * (close[i] - kama[-1]))
    kama = np.array(kama)
    
    # RSI(14)
    delta = close_s.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # Choppiness Index(14)
    atr = np.zeros(n-1)
    for i in range(n-1):
        tr1 = high[i+1] - low[i+1]
        tr2 = abs(high[i+1] - close[i])
        tr3 = abs(low[i+1] - close[i])
        atr[i] = max(tr1, tr2, tr3)
    atr_sum = pd.Series(atr).rolling(14, min_periods=14).sum()
    hh = pd.Series(high).rolling(14, min_periods=14).max()
    ll = pd.Series(low).rolling(14, min_periods=14).min()
    chop = 100 * np.log10(atr_sum / (hh - ll)) / np.log10(14)
    chop = chop.values
    chop = np.concatenate([np.full(13, np.nan), chop])
    
    # Weekly trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 100:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    weekly_uptrend = close_1w > ema200_1w
    weekly_downtrend = close_1w < ema200_1w
    
    # Align weekly trend to daily
    weekly_uptrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_uptrend.astype(float))
    weekly_downtrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_downtrend.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or
            np.isnan(weekly_uptrend_aligned[i]) or np.isnan(weekly_downtrend_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        kama_up = kama[i] > kama[i-1]
        kama_down = kama[i] < kama[i-1]
        rsi_oversold = rsi[i] < 40
        rsi_overbought = rsi[i] > 60
        chop_high = chop[i] > 61.8  # ranging market
        
        weekly_up = weekly_uptrend_aligned[i] > 0.5
        weekly_down = weekly_downtrend_aligned[i] > 0.5
        
        if position == 0:
            # Enter long: weekly uptrend + KAMA up + RSI oversold + choppy market
            if weekly_up and kama_up and rsi_oversold and chop_high:
                signals[i] = 0.25
                position = 1
            # Enter short: weekly downtrend + KAMA down + RSI overbought + choppy market
            elif weekly_down and kama_down and rsi_overbought and chop_high:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit conditions: trend changes or RSI normalizes
            if not weekly_up or not kama_up or rsi[i] > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit conditions: trend changes or RSI normalizes
            if not weekly_down or not kama_down or rsi[i] < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals