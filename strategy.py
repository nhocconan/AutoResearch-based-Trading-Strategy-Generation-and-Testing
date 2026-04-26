#!/usr/bin/env python3
"""
1d_KAMA_Trend_Filter_With_RSI_and_Chop_Regime_v1
Hypothesis: Daily KAMA (adaptive trend) provides robust trend direction with low whipsaw. 
Entry: KAMA trend + RSI(14) pullback (40-60 in uptrend, 40-60 in downtrend for short) + chop regime filter (CHOP > 61.8 = ranging) for mean reversion in ranges.
Exit: Opposite KAMA signal or RSI extreme (>70 long exit, <30 short exit).
Uses 1w EMA50 as HTF trend filter to avoid counter-trend trades in strong weekly trends.
Position size: 0.25 discrete to minimize fee churn. Target: 15-25 trades/year.
Works in bull/bear via adaptive trend and regime filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d and 1w data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # === 1d Indicators ===
    # KAMA (adaptive trend) - faster EMA in trend, slower in range
    change = np.abs(np.diff(close, 1))
    change = np.insert(change, 0, 0)
    volatility = np.sum(np.abs(np.diff(close, 1)), axis=0)  # placeholder, will compute correctly below
    # Correct volatility calculation: sum of abs changes over lookback
    er = np.zeros_like(close)
    for i in range(10, len(close)):
        if i >= 10:
            change_sum = np.sum(np.abs(np.diff(close[i-9:i+1])))
            volatility_sum = np.sum(np.abs(np.diff(close[i-9:i+1])))
            if volatility_sum > 0:
                er[i] = change_sum / volatility_sum
            else:
                er[i] = 0
    er[0:10] = 0
    sc = (er * (0.6667 - 0.0645) + 0.0645) ** 2  # smoothing constant
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    kama_aligned = kama  # already 1d, no alignment needed for same TF
    
    # RSI(14)
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Choppiness Index (14)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = high[0] - low[0]
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    hh14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr14 * 14 / np.maximum(hh14 - ll14, 1e-10)) / np.log10(14)
    
    # === 1w HTF Indicators ===
    # Weekly EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    # Weekly trend: 1 = above EMA50, -1 = below
    weekly_trend = np.where(close > ema_50_1w_aligned, 1, -1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 14 for RSI/chop, 50 for weekly EMA)
    start_idx = max(14, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(weekly_trend[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # KAMA trend: price above/below KAMA
        kama_trend = 1 if close[i] > kama[i] else -1
        
        if position == 0:
            # Long: KAMA uptrend + RSI in mid-range (40-60) + chop > 61.8 (ranging) + weekly trend alignment
            if (kama_trend == 1 and 40 <= rsi[i] <= 60 and chop[i] > 61.8 and weekly_trend[i] == 1):
                signals[i] = 0.25
                position = 1
            # Short: KAMA downtrend + RSI in mid-range (40-60) + chop > 61.8 (ranging) + weekly trend alignment
            elif (kama_trend == -1 and 40 <= rsi[i] <= 60 and chop[i] > 61.8 and weekly_trend[i] == -1):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: KAMA downtrend OR RSI > 70 (overbought) OR chop < 38.2 (strong trend - exit mean reversion)
            if (kama_trend == -1 or rsi[i] > 70 or chop[i] < 38.2):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: KAMA uptrend OR RSI < 30 (oversold) OR chop < 38.2 (strong trend - exit mean reversion)
            if (kama_trend == 1 or rsi[i] < 30 or chop[i] < 38.2):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_KAMA_Trend_Filter_With_RSI_and_Chop_Regime_v1"
timeframe = "1d"
leverage = 1.0