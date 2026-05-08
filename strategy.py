#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h KAMA + RSI + Chop Regime
# - KAMA adapts to market noise, effective in trending and ranging markets
# - RSI filters overbought/oversold conditions
# - Choppy market filter (Chop > 61.8) enables mean reversion; Chop < 38.2 enables trend following
# - Designed to work in both bull and bear markets via regime adaptation
# - Target: 20-35 trades/year to minimize fee drag on 4h timeframe

name = "4h_KAMA_RSI_Chop_Regime"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for Chop regime (using ATR-based Chop calculation)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range for Chop calculation
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with index
    
    # ATR(14) and Chop calculation
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    sum_tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    max_h = np.maximum.accumulate(high_1d)
    min_l = np.minimum.accumulate(low_1d)
    range_14 = max_h - min_l
    chop = np.where(range_14 != 0, 100 * np.log10(sum_tr_14 / range_14) / np.log10(14), 50)
    chop = np.where(np.isnan(range_14) | (range_14 == 0), 50, chop)
    
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # 4h KAMA calculation
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.concatenate([[np.nan], np.diff(close[:-1])]))
    for i in range(1, len(change)):
        change[i] = np.abs(close[i] - close[i-10]) if i >= 10 else np.nan
    change = pd.Series(change).rolling(window=10, min_periods=10).sum().values
    volatility = np.abs(np.concatenate([[np.nan], np.diff(close[:-1])]))
    volatility = pd.Series(volatility).rolling(window=10, min_periods=10).sum().values
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (0.6645 - 0.0645) + 0.0645) ** 2
    # KAMA calculation
    kama = np.full_like(close, np.nan)
    kama[9] = close[9]  # seed
    for i in range(10, len(close)):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # 4h RSI(14)
    delta = np.concatenate([[np.nan], np.diff(close[:-1])])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # ensure sufficient warmup for all indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price > KAMA + RSI < 50 (not overbought) + Chop regime filter
            # In trending markets (Chop < 38.2): follow trend (price > KAMA)
            # In ranging markets (Chop > 61.8): mean revert at extremes (RSI < 30)
            if chop_aligned[i] < 38.2:  # trending regime
                long_cond = (close[i] > kama[i] and rsi[i] < 50)
            elif chop_aligned[i] > 61.8:  # ranging regime
                long_cond = (rsi[i] < 30)
            else:  # transitional regime
                long_cond = (close[i] > kama[i] and rsi[i] < 40)
            
            # Short conditions: price < KAMA + RSI > 50 (not oversold) + Chop regime filter
            if chop_aligned[i] < 38.2:  # trending regime
                short_cond = (close[i] < kama[i] and rsi[i] > 50)
            elif chop_aligned[i] > 61.8:  # ranging regime
                short_cond = (rsi[i] > 70)
            else:  # transitional regime
                short_cond = (close[i] < kama[i] and rsi[i] > 60)
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price < KAMA OR RSI > 70 (overbought)
            if close[i] < kama[i] or rsi[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price > KAMA OR RSI < 30 (oversold)
            if close[i] > kama[i] or rsi[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals