#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 1d KAMA + RSI + Chop Filter with 1h Entry Timing
# Hypothesis: KAMA identifies trend direction on daily timeframe. RSI(14) provides
# entry timing on 1-hour timeframe during pullbacks. Chop filter (Choppiness Index)
# avoids whipsaw in ranging markets. This combination works in both bull and bear
# markets by focusing on strong trends with precise entries.
# Target: 10-25 trades/year to minimize fee drag on daily timeframe.
name = "1d_kama_rsi_chop_filter_v2"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Get weekly data for chop filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # KAMA on daily timeframe
    # Efficiency Ratio (ER)
    daily_close = df_1d['close'].values
    change = np.abs(np.diff(daily_close, k=10))  # 10-period change
    volatility = np.sum(np.abs(np.diff(daily_close)), axis=1)  # 10-period volatility
    er = np.divide(change, volatility, out=np.zeros_like(change), where=volatility!=0)
    # Smoothing constants
    sc = (er * (2/(2+2) - 2/(30+2)) + 2/(30+2)) ** 2  # fast=2, slow=30
    kama = np.full_like(daily_close, np.nan)
    kama[29] = daily_close[29]  # start after 30 periods
    for i in range(30, len(daily_close)):
        kama[i] = kama[i-1] + sc[i-1] * (daily_close[i] - kama[i-1])
    kama_1h = align_htf_to_ltf(prices, df_1d, kama)
    
    # RSI(14) on 1-hour timeframe
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    gain_ma = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean().values
    loss_ma = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean().values
    rs = np.divide(gain_ma, loss_ma, out=np.zeros_like(gain_ma), where=loss_ma!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Chop filter on weekly timeframe (Choppiness Index)
    # True Range
    tr1 = df_1w['high'] - df_1w['low']
    tr2 = np.abs(df_1w['high'] - np.append(df_1w['close'][0], df_1w['close'][:-1].values))
    tr3 = np.abs(df_1w['low'] - np.append(df_1w['close'][0], df_1w['close'][:-1].values))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    # Sum of TR over 14 periods
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    # Highest high and lowest low over 14 periods
    hh = df_1w['high'].rolling(window=14, min_periods=14).max().values
    ll = df_1w['low'].rolling(window=14, min_periods=14).min().values
    # Chop = 100 * log10(sum(tr)/(hh-ll)) / log10(14)
    chop = 100 * np.log10(atr_sum / (hh - ll + 1e-10)) / np.log10(14)
    chop_1h = align_htf_to_ltf(prices, df_1w, chop)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if required data not available
        if (np.isnan(kama_1h[i]) or np.isnan(rsi[i]) or np.isnan(chop_1h[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price below KAMA or RSI overbought
            if close[i] < kama_1h[i] or rsi[i] > 70:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit: price above KAMA or RSI oversold
            if close[i] > kama_1h[i] or rsi[i] < 30:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Chop filter: only trade when trending (CHOP < 38.2)
            if chop_1h[i] < 38.2:
                # Long: price above KAMA + RSI pulling back from oversold
                if close[i] > kama_1h[i] and 30 < rsi[i] < 50:
                    position = 1
                    signals[i] = 0.25
                # Short: price below KAMA + RSI pulling back from overbought
                elif close[i] < kama_1h[i] and 50 < rsi[i] < 70:
                    position = -1
                    signals[i] = -0.25
    
    return signals