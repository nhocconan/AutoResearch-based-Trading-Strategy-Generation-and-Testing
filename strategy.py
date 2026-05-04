#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA trend with RSI(14) mean reversion entries and choppiness regime filter
# Uses 1-week EMA34 for higher timeframe trend alignment (more stable than 1d in ranging markets)
# KAMA adapts to market noise, reducing whipsaw in choppy conditions
# RSI(14) < 30 for longs, > 70 for shorts in trending markets (chop < 61.8)
# Discrete sizing 0.25 limits risk and reduces fee churn. Target: 50-100 trades over 4 years.

name = "1d_KAMA_RSI_ChopRegime"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate 1w EMA34 for trend direction
    close_1w = pd.Series(df_1w['close'])
    ema34_1w = close_1w.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1w EMA34 to 1d timeframe (completed 1w bar only)
    ema34_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Calculate KAMA(10,2,30) on 1d timeframe
    close_s = pd.Series(close)
    direction = abs(close_s.diff(10))
    volatility = close_s.diff(1).abs().rolling(window=10, min_periods=1).sum()
    er = direction / volatility.replace(0, np.nan)
    er = er.fillna(0).clip(0, 1)
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc.iloc[i] * (close[i] - kama[i-1])
    
    # Calculate RSI(14) on 1d timeframe
    delta = close_s.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    
    # Calculate Choppiness Index(14) on 1d timeframe
    atr = np.zeros(n)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0]-low[0], np.abs(high[0]-close[0]), np.abs(low[0]-close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    sum_tr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    max_h14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_l14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(sum_tr14 / (max_h14 - min_l14)) / np.log10(14)
    chop = np.nan_to_num(chop, nan=50.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema34_aligned[i]) or np.isnan(kama[i]) or 
            np.isnan(rsi[i]) or np.isnan(chop[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Regime filter: choppiness < 61.8 = trending (favor trend following)
        trending_regime = chop[i] < 61.8
        
        if position == 0:
            # Long conditions: price > KAMA + uptrend + RSI oversold + trending regime
            if close[i] > kama[i] and close[i] > ema34_aligned[i] and rsi[i] < 30 and trending_regime:
                signals[i] = 0.25
                position = 1
            # Short conditions: price < KAMA + downtrend + RSI overbought + trending regime
            elif close[i] < kama[i] and close[i] < ema34_aligned[i] and rsi[i] > 70 and trending_regime:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price < KAMA OR RSI > 50 OR regime changes to choppy
            if close[i] < kama[i] or rsi[i] > 50 or chop[i] >= 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price > KAMA OR RSI < 50 OR regime changes to choppy
            if close[i] > kama[i] or rsi[i] < 50 or chop[i] >= 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals