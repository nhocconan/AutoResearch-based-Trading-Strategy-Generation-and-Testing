#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA trend + RSI(14) extremes + chop regime filter (CHOP > 61.8 = range)
# Long when KAMA up AND RSI < 30 (oversold) AND choppy market (CHOP > 61.8)
# Short when KAMA down AND RSI > 70 (overbought) AND choppy market (CHOP > 61.8)
# Uses 1w EMA50 as HTF trend filter: only take longs when price > 1w EMA50 in bull regime,
# only shorts when price < 1w EMA50 in bear regime. Adaptive to market conditions.
# KAMA adapts to market noise, reducing whipsaw in ranging markets.
# RSI extremes provide mean reversion entries in chop.
# Chop regime filter ensures we only mean revert in ranging markets, avoiding trends.
# Target: 20-60 total trades over 4 years (5-15/year) to minimize fee drag.
# Works in bull (trend filter allows momentum) and bear (mean reversion in chop).

name = "1d_KAMA_RSI_ChopRegime_1wEMA50_TrendFilter"
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
    
    # Get 1d data ONCE before loop for KAMA, RSI, and chop calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Get 1w data ONCE before loop for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1d KAMA (adaptive moving average)
    close_1d = df_1d['close'].values
    # Efficiency ratio (ER) over 10 periods
    change = np.abs(np.diff(close_1d, n=10))  # 10-period net change
    volatility = np.sum(np.abs(np.diff(close_1d, n=1)), axis=0)  # 10-period sum of absolute changes
    # Handle first 10 values where diff returns empty
    change_full = np.concatenate([np.full(10, np.nan), change])
    volatility_full = np.concatenate([np.full(10, np.nan), volatility])
    er = np.where(volatility_full > 0, change_full / volatility_full, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    # Calculate KAMA
    kama = np.full_like(close_1d, np.nan)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    # Calculate 1d RSI(14)
    delta = np.diff(close_1d)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    # First average gain/loss
    avg_gain = np.concatenate([np.full(14, np.nan), [np.mean(gain[:14])] if len(gain) >= 14 else [np.nan]])
    avg_loss = np.concatenate([np.full(14, np.nan), [np.mean(loss[:14])] if len(loss) >= 14 else [np.nan]])
    # Wilder's smoothing
    for i in range(15, len(gain)+1):
        if i < len(avg_gain):
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i-1]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i-1]) / 14
    rs = np.where(avg_loss > 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # Calculate 1d Choppiness Index (CHOP) over 14 periods
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])  # align with close
    # ATR(14)
    atr = np.zeros_like(close_1d)
    atr[13] = np.nanmean(tr[1:15]) if len(tr) >= 15 else np.nan
    for i in range(14, len(atr)):
        atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    # Sum of ATR over 14 periods
    sum_atr_14 = np.zeros_like(close_1d)
    for i in range(13, len(sum_atr_14)):
        if i >= 13:
            sum_atr_14[i] = np.sum(atr[i-13:i+1])
    # Max(high) - Min(low) over 14 periods
    max_high = np.zeros_like(close_1d)
    min_low = np.zeros_like(close_1d)
    for i in range(len(close_1d)):
        start_idx = max(0, i-13)
        max_high[i] = np.max(high[start_idx:i+1])
        min_low[i] = np.min(low[start_idx:i+1])
    # Chop = 100 * log10(sum(ATR14) / (maxH - minL)) / log10(14)
    range_14 = max_high - min_low
    chop = np.zeros_like(close_1d)
    mask = (range_14 > 0) & (sum_atr_14 > 0)
    chop[mask] = 100 * np.log10(sum_atr_14[mask] / range_14[mask]) / np.log10(14)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Calculate 1w EMA50 trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(chop_aligned[i]) or np.isnan(ema_50_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: KAMA up (price > KAMA) AND RSI < 30 (oversold) AND choppy (CHOP > 61.8) AND price > 1w EMA50 (bull filter)
            if (close[i] > kama_aligned[i] and 
                rsi_aligned[i] < 30 and 
                chop_aligned[i] > 61.8 and 
                close[i] > ema_50_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: KAMA down (price < KAMA) AND RSI > 70 (overbought) AND choppy (CHOP > 61.8) AND price < 1w EMA50 (bear filter)
            elif (close[i] < kama_aligned[i] and 
                  rsi_aligned[i] > 70 and 
                  chop_aligned[i] > 61.8 and 
                  close[i] < ema_50_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: KAMA down OR RSI > 50 (exit mean reversion) OR chop < 50 (trending market)
            if (close[i] < kama_aligned[i] or 
                rsi_aligned[i] > 50 or 
                chop_aligned[i] < 50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: KAMA up OR RSI < 50 (exit mean reversion) OR chop < 50 (trending market)
            if (close[i] > kama_aligned[i] or 
                rsi_aligned[i] < 50 or 
                chop_aligned[i] < 50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals