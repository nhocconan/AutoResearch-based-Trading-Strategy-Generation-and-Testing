#!/usr/bin/env python3
"""
1d_KAMA_Trend_RSI_ChopFilter_RegimeAdaptive
Hypothesis: On daily timeframe, use Kaufman Adaptive Moving Average (KAMA) for trend direction,
combined with RSI(14) for momentum confirmation and Choppiness Index (CHOP) for regime filtering.
Enter long when KAMA slope > 0, RSI > 50, and CHOP < 61.8 (trending regime).
Enter short when KAMA slope < 0, RSI < 50, and CHOP < 61.8 (trending regime).
Exit when any condition fails or CHOP > 61.8 (range regime triggers mean-reversion avoidance).
Uses 1w EMA50 as higher timeframe trend filter to avoid counter-trend trades in strong weekly trends.
Discrete sizing (0.25) minimizes fee drag. Target: 30-100 trades over 4 years.
Designed to work in bull markets via trend following and in bear markets by avoiding range-bound whipsaws
via CHOP regime filter and weekly EMA trend alignment.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === KAMA Calculation (Primary Trend Indicator) ===
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close, n=10))  # |close[t] - close[t-10]|
    volatility = np.sum(np.abs(np.diff(close)), axis=1)  # Sum of |close[t] - close[t-1]| over 10 periods
    # Avoid division by zero
    er = np.zeros_like(close)
    er[10:] = change[10:] / np.where(volatility[10:] == 0, 1, volatility[10:])
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    # Calculate KAMA
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # KAMA slope (trend direction)
    kama_slope = np.diff(kama, n=1)
    kama_slope = np.append(kama_slope, kama_slope[-1])  # same length as close
    
    # === RSI(14) Calculation ===
    delta = np.diff(close, n=1)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing (equivalent to EMA with alpha=1/14)
    avg_gain = np.zeros_like(close)
    avg_loss = np.zeros_like(close)
    avg_gain[13] = np.mean(gain[1:14])  # first average of first 14 gains
    avg_loss[13] = np.mean(loss[1:14])  # first average of first 14 losses
    
    for i in range(14, n):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.zeros_like(close)
    rs[13:] = avg_gain[13:] / np.where(avg_loss[13:] == 0, 1, avg_loss[13:])
    rsi = 100 - (100 / (1 + rs))
    # For first 13 periods, set to 50 (neutral)
    rsi[:13] = 50
    
    # === Choppiness Index (CHOP) Calculation ===
    # True Range over 14 periods
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period has no previous close
    
    atr14 = np.zeros_like(close)
    for i in range(14, n):
        atr14[i] = np.sum(tr[i-13:i+1])  # sum of last 14 TR values
    # For first 14 periods, use cumulative sum
    for i in range(1, 14):
        atr14[i] = np.sum(tr[0:i+1])
    
    # Highest high and lowest low over 14 periods
    max_high = np.zeros_like(close)
    min_low = np.zeros_like(close)
    for i in range(14, n):
        max_high[i] = np.max(high[i-13:i+1])
        min_low[i] = np.min(low[i-13:i+1])
    # For first 14 periods
    for i in range(1, 14):
        max_high[i] = np.max(high[0:i+1])
        min_low[i] = np.min(low[0:i+1])
    max_high[0] = high[0]
    min_low[0] = low[0]
    
    # CHOP = 100 * log10(sum(TR14) / (HH14 - LL14)) / log10(14)
    range_hl = max_high - min_low
    # Avoid division by zero and log of zero
    chop = np.zeros_like(close)
    for i in range(14, n):
        if range_hl[i] > 0:
            chop[i] = 100 * np.log10(atr14[i] / range_hl[i]) / np.log10(14)
        else:
            chop[i] = 0  # no range, set to 0 (will be < 61.8)
    # For first 14 periods, set to 50 (neutral)
    chop[:14] = 50
    
    # === Higher Timeframe Filter: 1w EMA50 ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Start after warmup (need 30 for KAMA stability, 14 for RSI/CHOP, 50 for 1w EMA)
    start_idx = max(30, 14, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama_slope[i]) or np.isnan(rsi[i]) or 
            np.isnan(chop[i]) or np.isnan(ema_50_1w_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Regime filter: only trade in trending regime (CHOP < 61.8)
        trending_regime = chop[i] < 61.8
        
        # Long logic: KAMA up, RSI > 50, trending regime, and price above weekly EMA (bullish bias)
        long_condition = (kama_slope[i] > 0) and (rsi[i] > 50) and trending_regime and (close[i] > ema_50_1w_aligned[i])
        # Short logic: KAMA down, RSI < 50, trending regime, and price below weekly EMA (bearish bias)
        short_condition = (kama_slope[i] < 0) and (rsi[i] < 50) and trending_regime and (close[i] < ema_50_1w_aligned[i])
        
        # Exit logic: any condition fails
        exit_long = not ((kama_slope[i] > 0) and (rsi[i] > 50) and trending_regime)
        exit_short = not ((kama_slope[i] < 0) and (rsi[i] < 50) and trending_regime)
        
        if long_condition and position != 1:
            signals[i] = base_size
            position = 1
        elif short_condition and position != -1:
            signals[i] = -base_size
            position = -1
        elif position == 1 and exit_long:
            signals[i] = 0.0
            position = 0
        elif position == -1 and exit_short:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
    
    return signals

name = "1d_KAMA_Trend_RSI_ChopFilter_RegimeAdaptive"
timeframe = "1d"
leverage = 1.0