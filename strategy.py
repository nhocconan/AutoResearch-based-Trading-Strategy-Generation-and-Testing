#!/usr/bin/env python3
# 1d_KAMA_Direction_RSI_ChopFilter_v2
# Hypothesis: 1d KAMA trend direction combined with RSI extremes and Choppiness index regime filter.
# KAMA adapts to market noise, reducing whipsaws in sideways markets. RSI identifies overbought/oversold conditions.
# Choppiness index filters for trending regimes (CHOP < 38.2) to avoid false signals in ranging markets.
# Works in bull markets via RSI oversold bounces in uptrend, and in bear markets via RSI overbought reversals in downtrend.
# Uses 1d timeframe to minimize trade frequency (target: 30-100 total over 4 years) and reduce fee drag.

name = "1d_KAMA_Direction_RSI_ChopFilter_v2"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1w data for trend filter (more stable than 1d EMA)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1d OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # KAMA calculation (ER = 10, fast = 2, slow = 30)
    close_s = pd.Series(close)
    change = abs(close_s - close_s.shift(10))
    volatility = abs(close_s.diff()).rolling(window=10, min_periods=1).sum()
    er = change / volatility.replace(0, 1e-10)
    sc = (er * (2/2 - 2/30) + 2/30) ** 2
    kama = [close[0]]
    for i in range(1, len(close)):
        kama.append(kama[-1] + sc.iloc[i] * (close[i] - kama[-1]))
    kama = np.array(kama)
    
    # RSI (14)
    delta = pd.Series(close).diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=14).mean()
    rs = gain / loss.replace(0, 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # Choppiness Index (14)
    atr = pd.Series(np.maximum(high - low, np.maximum(high - close_s.shift(), low - close_s.shift()))).rolling(window=14, min_periods=14).mean()
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max()
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min()
    chop = 100 * np.log10((atr * np.sqrt(14)) / (max_high - min_low)) / np.log10(14)
    chop = chop.values
    
    # Align 1w trend filter (close > EMA50)
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    close_1w_aligned = align_htf_to_ltf(prices, df_1w, df_1w['close'].values)
    uptrend = close_1w_aligned > ema_50_1w_aligned
    downtrend = close_1w_aligned < ema_50_1w_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need enough data for KAMA, RSI, CHOP, and 1w EMA
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(kama[i]) or
            np.isnan(rsi[i]) or
            np.isnan(chop[i]) or
            np.isnan(ema_50_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: KAMA bullish (price > KAMA) + RSI oversold (<30) + trending regime (CHOP < 38.2)
            if close[i] > kama[i] and rsi[i] < 30 and chop[i] < 38.2:
                signals[i] = 0.25
                position = 1
            # Short: KAMA bearish (price < KAMA) + RSI overbought (>70) + trending regime (CHOP < 38.2)
            elif close[i] < kama[i] and rsi[i] > 70 and chop[i] < 38.2:
                signals[i] = -0.25
                position = -1
        else:
            if position == 1:
                # Long exit: KAMA bearish crossover OR RSI overbought (>70)
                if close[i] < kama[i] or rsi[i] > 70:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Short exit: KAMA bullish crossover OR RSI oversold (<30)
                if close[i] > kama[i] or rsi[i] < 30:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals