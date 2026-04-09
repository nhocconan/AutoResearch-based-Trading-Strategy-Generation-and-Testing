#!/usr/bin/env python3
# 1d_kama_rsi_chop_v1
# Hypothesis: Daily KAMA trend direction + RSI extremes + chop regime filter.
# KAMA adapts to market noise, RSI captures overbought/oversold, chop filter avoids whipsaws in ranging markets.
# Works in bull/bear: KAMA trend + RSI mean reversion with chop filter reduces false signals.
# Target: 10-25 trades/year.

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
    
    # 1d KAMA(14, ER=10)
    close_s = pd.Series(close)
    change = abs(close_s.diff(1))
    volatility = change.rolling(window=10, min_periods=10).sum()
    er = change.rolling(window=10, min_periods=10).sum() / volatility.replace(0, np.nan)
    sc = (er * (0.6 - 0.06) + 0.06) ** 2
    kama = pd.Series(index=close_s.index, dtype=float)
    kama.iloc[0] = close_s.iloc[0]
    for i in range(1, len(close_s)):
        kama.iloc[i] = kama.iloc[i-1] + sc.iloc[i] * (close_s.iloc[i] - kama.iloc[i-1])
    kama = kama.values
    
    # 1d RSI(14)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # 1d Chopiness Index(14)
    atr = pd.Series(np.maximum(high - low, np.maximum(high - close_s.shift(1), low - close_s.shift(1)))).rolling(window=14, min_periods=14).mean()
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max()
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min()
    chop = 100 * np.log10(atr.sum() / (highest_high - lowest_low)) / np.log10(14)
    chop = chop.values
    
    # 1w HTF EMA(21) for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 21:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=21, min_periods=21, adjust=False).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or np.isnan(ema_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: trend turns bearish OR RSI > 70 (overbought)
            if close[i] < kama[i] or rsi[i] > 70:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: trend turns bullish OR RSI < 30 (oversold)
            if close[i] > kama[i] or rsi[i] < 30:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Chop regime filter: only trade when chop > 61.8 (ranging market)
            if chop[i] > 61.8:
                # Mean reversion: buy oversold, sell overbought
                if rsi[i] < 30 and close[i] > kama[i]:
                    # Oversold + price above KAMA → long
                    position = 1
                    signals[i] = 0.25
                elif rsi[i] > 70 and close[i] < kama[i]:
                    # Overbought + price below KAMA → short
                    position = -1
                    signals[i] = -0.25
    
    return signals