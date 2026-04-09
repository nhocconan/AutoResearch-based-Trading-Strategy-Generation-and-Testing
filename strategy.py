#!/usr/bin/env python3
# 1d_kama_rsi_chop_v1
# Hypothesis: 1d strategy using KAMA direction for trend, RSI(14) for momentum/extremes,
# and Choppiness Index regime filter to avoid whipsaws. Long when KAMA up, RSI<30, CHOP>61.8 (range).
# Short when KAMA down, RSI>70, CHOP>61.8. Uses 1w HTF for trend filter: only long when price>weekly EMA200.
# Discrete sizing 0.0, ±0.25 to minimize fee churn. Target: 10-20 trades/year.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_kama_rsi_chop_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1w HTF data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    weekly_ema200 = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    weekly_ema200_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema200)
    
    # KAMA direction (ER=10, fast=2, slow=30)
    close_s = pd.Series(close)
    change = abs(close_s.diff(10))
    volatility = close_s.diff().abs().rolling(window=10, min_periods=10).sum()
    er = change / volatility.replace(0, np.nan)
    er = er.fillna(0).clip(0, 1).values
    sc = (er * (2/2 - 2/30) + 2/30) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    kama_dir = np.where(kama > np.roll(kama, 1), 1, -1)
    kama_dir[0] = 0
    
    # RSI(14)
    delta = close_s.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    
    # Choppiness Index (14)
    atr = pd.Series(np.maximum(high - low, np.maximum(high - np.roll(close, 1), np.roll(close, 1) - low))).rolling(window=14, min_periods=14).mean().values
    hh = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr.sum() / (hh - ll)) / np.log10(14)
    chop = chop.values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(200, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(kama_dir[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or
            np.isnan(weekly_ema200_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: only trade in ranging markets (CHOP > 61.8)
        if chop[i] <= 61.8:
            # Exit positions when not ranging
            if position != 0:
                position = 0
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: KAMA turns down OR RSI > 50 (exit extreme)
            if kama_dir[i] == -1 or rsi[i] > 50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: KAMA turns up OR RSI < 50 (exit extreme)
            if kama_dir[i] == 1 or rsi[i] < 50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Only consider entries when price > weekly EMA200 (bullish bias from 1w)
            if close[i] > weekly_ema200_aligned[i]:
                # Long entry: KAMA up AND RSI < 30 (oversold in range)
                if kama_dir[i] == 1 and rsi[i] < 30:
                    position = 1
                    signals[i] = 0.25
            else:
                # Short entry: KAMA down AND RSI > 70 (overbought in range)
                if kama_dir[i] == -1 and rsi[i] > 70:
                    position = -1
                    signals[i] = -0.25
    
    return signals