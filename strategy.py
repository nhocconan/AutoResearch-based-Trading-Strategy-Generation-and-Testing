#!/usr/bin/env python3
"""
1d_KAMA_Trend_RSI_ChopFilter_v1
Hypothesis: On daily timeframe, use Kaufman Adaptive Moving Average (KAMA) for trend direction, RSI(14) for momentum confirmation, and Choppiness Index(14) for regime filtering. Enter long when KAMA slopes up, RSI > 50, and market is trending (CHOP < 38.2). Enter short when KAMA slopes down, RSI < 50, and CHOP < 38.2. Exit on opposite signal. Uses 1w EMA200 as higher timeframe trend filter to avoid counter-trend trades in strong weekly trends. Designed for low trade frequency (~10-20/year) to minimize fee drag and work in both bull and bear markets by adapting to trending regimes only.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop: 1w for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # === 1w EMA200 for higher timeframe trend filter ===
    close_1w = df_1w['close'].values
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # === Daily KAMA (10, 2, 30) ===
    # ER = Efficiency Ratio, SC = Smoothing Constant
    close = prices['close'].values
    direction = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close, prepend=close[0])), axis=0) if False else None  # placeholder
    # Correct ER calculation over 10 periods
    er = np.zeros_like(close)
    for i in range(10, n):
        if i >= 10:
            net_change = abs(close[i] - close[i-10])
            total_change = np.sum(np.abs(np.diff(close[i-10:i+1])))
            er[i] = net_change / total_change if total_change != 0 else 0
    # SC = [ER * (fastest - slowest) + slowest]^2
    fastest = 2.0 / (2 + 1)
    slowest = 2.0 / (30 + 1)
    sc = (er * (fastest - slowest) + slowest) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    kama_aligned = kama  # already LTF
    
    # === Daily RSI(14) ===
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi_aligned = rsi  # already LTF
    
    # === Daily Choppiness Index(14) ===
    high = prices['high'].values
    low = prices['low'].values
    atr = np.zeros_like(close)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = np.zeros_like(close)
    for i in range(14, n):
        atr_sum = np.sum(atr[i-13:i+1])
        hhll = highest_high[i] - lowest_low[i]
        chop[i] = 100 * np.log10(atr_sum / hhll) / np.log10(14) if hhll != 0 else 50
    chop_aligned = chop  # already LTF
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(ema_200_1w_aligned[i]) or
            np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or np.isnan(chop_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = close[i]
        kama_now = kama_aligned[i]
        kama_prev = kama_aligned[i-1]
        rsi_now = rsi_aligned[i]
        chop_now = chop_aligned[i]
        weekly_trend = ema_200_1w_aligned[i]
        
        # KAMA slope: rising if today > yesterday
        kama_rising = kama_now > kama_prev
        kama_falling = kama_now < kama_prev
        
        # Regime filter: only trade in trending markets (CHOP < 38.2)
        trending_regime = chop_now < 38.2
        
        if position == 0:
            # Long: KAMA rising, RSI > 50, trending regime, price above weekly EMA200
            if kama_rising and rsi_now > 50 and trending_regime and price_close > weekly_trend:
                signals[i] = 0.25
                position = 1
            # Short: KAMA falling, RSI < 50, trending regime, price below weekly EMA200
            elif kama_falling and rsi_now < 50 and trending_regime and price_close < weekly_trend:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit on opposite signal
            if position == 1 and (not kama_rising or rsi_now < 50):
                signals[i] = 0.0
                position = 0
            elif position == -1 and (not kama_falling or rsi_now > 50):
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_KAMA_Trend_RSI_ChopFilter_v1"
timeframe = "1d"
leverage = 1.0