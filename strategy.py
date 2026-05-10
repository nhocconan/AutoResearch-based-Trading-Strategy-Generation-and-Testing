#!/usr/bin/env python3
# 1d_KAMA_RSI_ChopFilter
# Hypothesis: KAMA identifies trend direction with adaptive smoothing, RSI(14) filters extremes, and Choppiness Index avoids ranging markets.
# In trending markets (CHOP < 38.2), go long when KAMA upward and RSI > 50, short when KAMA downward and RSI < 50.
# Works in bull/bear by following adaptive trend. Targets 15-25 trades/year to minimize fee drag on 1d timeframe.

name = "1d_KAMA_RSI_ChopFilter"
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
    volume = prices['volume'].values
    
    # KAMA calculation
    close_series = pd.Series(close)
    change = abs(close_series.diff(1))
    volatility = change.rolling(window=10, min_periods=10).sum()
    er = change.rolling(window=10, min_periods=10).sum() / volatility.replace(0, 1e-10)
    sc = (er * (0.6645 - 0.0645) + 0.0645) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI(14)
    delta = close_series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    
    # Choppiness Index (14)
    atr1 = np.maximum(high - low, np.absolute(np.maximum(high - np.roll(close, 1), np.roll(low, 1) - low)))
    atr1_sum = pd.Series(atr1).rolling(window=14, min_periods=14).sum()
    hh14 = pd.Series(high).rolling(window=14, min_periods=14).max()
    ll14 = pd.Series(low).rolling(window=14, min_periods=14).min()
    chop = 100 * np.log10(atr1_sum / (hh14 - ll14)) / np.log10(14)
    chop = chop.fillna(50).values
    
    # Weekly trend filter (EMA34)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    ema_34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(10, 14, 14) + 1  # Warmup for KAMA/RSI/CHOP
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or 
            np.isnan(ema_34_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Weekly trend filter
        weekly_uptrend = close[i] > ema_34_1w_aligned[i]
        weekly_downtrend = close[i] < ema_34_1w_aligned[i]
        
        # Choppiness regime filter: only trade when trending (CHOP < 38.2)
        trending_regime = chop[i] < 38.2
        
        if position == 0:
            # Long entry: KAMA up, RSI > 50, weekly uptrend, trending regime
            if kama[i] > kama[i-1] and rsi[i] > 50 and weekly_uptrend and trending_regime:
                signals[i] = 0.25
                position = 1
            # Short entry: KAMA down, RSI < 50, weekly downtrend, trending regime
            elif kama[i] < kama[i-1] and rsi[i] < 50 and weekly_downtrend and trending_regime:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: KAMA down OR RSI < 50 OR ranging market
            if kama[i] < kama[i-1] or rsi[i] < 50 or chop[i] >= 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: KAMA up OR RSI > 50 OR ranging market
            if kama[i] > kama[i-1] or rsi[i] > 50 or chop[i] >= 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals