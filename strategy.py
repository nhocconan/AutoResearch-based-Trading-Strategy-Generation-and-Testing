#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 1d Weekly KAMA + RSI + Chop Filter
# Hypothesis: KAMA (Kaufman Adaptive Moving Average) adapts to market noise, providing
# reliable trend direction in both trending and ranging markets. Combined with weekly
# trend filter, RSI for momentum confirmation, and Choppiness Index to avoid whipsaws
# in high-noise regimes, this strategy captures sustained moves while minimizing false
# signals. Designed for low-frequency trading (7-25 trades/year) to reduce fee drag.

name = "1d_kama_rsi_chop_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # KAMA (Kaufman Adaptive Moving Average) - 14-period
    # Efficiency Ratio (ER) = |change| / sum(|changes|)
    change = np.abs(np.diff(close, prepend=close[0]))
    abs_change = np.abs(np.diff(close, prepend=close[0]))
    er = np.zeros(n)
    for i in range(1, n):
        if i < 10:
            er[i] = np.nan
        else:
            sum_abs_change = np.sum(abs_change[i-9:i+1])
            if sum_abs_change > 0:
                er[i] = change[i] / sum_abs_change
            else:
                er[i] = 0
    # Smoothing constants
    fast_sc = 2 / (2 + 1)  # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    kama = np.full(n, np.nan)
    kama[0] = close[0]
    for i in range(1, n):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # Weekly trend filter: EMA20 of weekly close
    ema_20_1w = pd.Series(df_1w['close']).ewm(span=20, adjust=False).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # RSI(14) - momentum confirmation
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, 50.0), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Choppiness Index (14) - regime filter
    # Chop = 100 * log10(sum(ATR) / (max(high) - min(low))) / log10(n)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = np.full(n, 50.0)
    for i in range(14, n):
        if max_high[i] - min_low[i] > 0:
            chop[i] = 100 * np.log10(atr_sum[i] / (max_high[i] - min_low[i])) / np.log10(14)
        else:
            chop[i] = 50.0
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(kama[i]) or np.isnan(ema_20_1w_aligned[i]) or 
            np.isnan(rsi[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend: price vs weekly EMA20
        if close[i] > ema_20_1w_aligned[i]:  # Uptrend
            # Long conditions: price > KAMA AND RSI > 50 AND Chop < 61.8 (trending)
            if close[i] > kama[i] and rsi[i] > 50 and chop[i] < 61.8:
                if position != 1:
                    position = 1
                    signals[i] = 0.25
                else:
                    signals[i] = 0.25
            # Exit long: price < KAMA OR RSI < 40 OR Chop > 61.8
            elif position == 1:
                if close[i] < kama[i] or rsi[i] < 40 or chop[i] > 61.8:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:
                signals[i] = 0.0
        else:  # Downtrend
            # Short conditions: price < KAMA AND RSI < 50 AND Chop < 61.8 (trending)
            if close[i] < kama[i] and rsi[i] < 50 and chop[i] < 61.8:
                if position != -1:
                    position = -1
                    signals[i] = -0.25
                else:
                    signals[i] = -0.25
            # Exit short: price > KAMA OR RSI > 60 OR Chop > 61.8
            elif position == -1:
                if close[i] > kama[i] or rsi[i] > 60 or chop[i] > 61.8:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals