#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA trend + RSI(14) mean reversion + chop regime filter
# KAMA adapts to market noise, providing reliable trend direction in both bull and bear markets.
# RSI(14) < 30 for long, > 70 for short provides mean reversion entries within the trend.
# Choppiness Index (CHOP) > 61.8 filters for ranging markets where mean reversion works best.
# Weekly HTF bias ensures alignment with higher timeframe structure.
# Works in bull (trend continuation on pullbacks) and bear (mean reversion at extremes in range).

name = "1d_KAMA_RSI_ChopRegime_WeeklyBias_v1"
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
    
    # Weekly HTF data for bias
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Weekly EMA(34) for trend bias
    close_1w = pd.Series(df_1w['close'].values)
    ema_34_1w = close_1w.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Daily KAMA(10, ER=10) - adaptive trend
    close_s = pd.Series(close)
    change = abs(close_s - close_s.shift(10))
    volatility = close_s.diff().abs().rolling(window=10, min_periods=10).sum()
    er = change / volatility.replace(0, np.nan)
    er = er.fillna(0)
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    sc = sc.fillna(0)
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Daily RSI(14)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    
    # Daily Choppiness Index(14)
    atr_14 = pd.Series(high - low).rolling(window=14, min_periods=14).mean()
    high_14 = pd.Series(high).rolling(window=14, min_periods=14).max()
    low_14 = pd.Series(low).rolling(window=14, min_periods=14).min()
    chop = 100 * np.log10(atr_14.rolling(window=14, min_periods=14).sum() / 
                          np.log10(high_14 - low_14)) / np.log10(14)
    chop = chop.fillna(50).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = 50
    
    for i in range(start_idx, n):
        if np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or np.isnan(ema_34_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_kama = kama[i]
        curr_rsi = rsi[i]
        curr_chop = chop[i]
        weekly_bias = ema_34_1w_aligned[i]
        
        # Regime filter: only trade in ranging markets (CHOP > 61.8)
        in_range = curr_chop > 61.8
        
        if position == 0:  # Flat - look for new entries
            # Long: price > KAMA (uptrend), RSI < 30 (oversold), in ranging market, above weekly EMA
            if curr_close > curr_kama and curr_rsi < 30 and in_range and curr_close > weekly_bias:
                signals[i] = 0.25
                position = 1
            # Short: price < KAMA (downtrend), RSI > 70 (overbought), in ranging market, below weekly EMA
            elif curr_close < curr_kama and curr_rsi > 70 and in_range and curr_close < weekly_bias:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on RSI > 50 (mean reversion complete) or trend change
            if curr_rsi > 50 or curr_close < curr_kama:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on RSI < 50 (mean reversion complete) or trend change
            if curr_rsi < 50 or curr_close > curr_kama:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals