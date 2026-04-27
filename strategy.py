#!/usr/bin/env python3
"""
1d_KAMA_RSI_Chop_Filter
Hypothesis: Uses KAMA direction (trend) on daily timeframe filtered by RSI(30-70 range) and Choppiness Index regime.
Only enters when trend is aligned and market is not too choppy (CHOP > 61.8 = range, < 38.2 = trending).
Designed for low trade frequency (~15-25 trades/year) to minimize fee drift while capturing medium-term trends.
Works in both bull and bear markets by avoiding whipsaws via regime filter.
"""

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
    volume = prices['volume'].values
    
    # KAMA (Kaufman Adaptive Moving Average) - 10 period
    # ER = Efficiency Ratio = |change| / sum(|changes|)
    change = np.abs(np.diff(close, prepend=close[0]))
    abs_change = np.abs(np.diff(close, prepend=close[0]))
    # Avoid division by zero
    er = np.divide(change, abs_change, out=np.zeros_like(change), where=abs_change!=0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1))**2  # fast=2, slow=30
    # KAMA calculation
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI (14-period)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Choppiness Index (14-period)
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = high[0] - low[0]
    tr2[0] = np.abs(high[0] - close[0])
    tr3[0] = np.abs(low[0] - close[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    # Sum of TR over period
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    # Highest high and lowest low over period
    hh = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low).rolling(window=14, min_periods=14).min().values
    # Choppy index
    chop = np.zeros(n)
    for i in range(n):
        if tr_sum[i] > 0 and hh[i] > ll[i]:
            chop[i] = 100 * np.log10(tr_sum[i] / (hh[i] - ll[i])) / np.log10(14)
        else:
            chop[i] = 50  # neutral when undefined
    
    # Weekly trend filter (1w close > 20-week EMA = uptrend)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for all indicators
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or 
            np.isnan(ema20_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Conditions
        price_above_kama = close[i] > kama[i]
        price_below_kama = close[i] < kama[i]
        rsi_mid = (rsi[i] >= 30) and (rsi[i] <= 70)  # Not overbought/oversold
        chop_range = chop[i] > 61.8  # Choppy market (range)
        chop_trend = chop[i] < 38.2  # Trending market
        weekly_uptrend = close[i] > ema20_1w_aligned[i]
        weekly_downtrend = close[i] < ema20_1w_aligned[i]
        
        if position == 0:
            # Long: price above KAMA, RSI in middle, trending market, weekly uptrend
            if price_above_kama and rsi_mid and chop_trend and weekly_uptrend:
                signals[i] = size
                position = 1
            # Short: price below KAMA, RSI in middle, trending market, weekly downtrend
            elif price_below_kama and rsi_mid and chop_trend and weekly_downtrend:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long: price below KAMA OR choppy market
            if price_below_kama or chop_range:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price above KAMA OR choppy market
            if price_above_kama or chop_range:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_KAMA_RSI_Chop_Filter"
timeframe = "1d"
leverage = 1.0