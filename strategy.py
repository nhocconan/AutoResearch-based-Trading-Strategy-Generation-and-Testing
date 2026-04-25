#!/usr/bin/env python3
"""
1d_KAMA_Trend_RSI_ChopFilter_v1
Hypothesis: On daily timeframe, use Kaufman Adaptive Moving Average (KAMA) for trend direction,
RSI(14) for momentum confirmation, and Choppiness Index(14) for regime filter.
Only trade when: 1) price > KAMA (uptrend) or price < KAMA (downtrend),
2) RSI confirms momentum (long: RSI>50, short: RSI<50),
3) Choppiness Index < 50 (trending market).
This avoids sideways chop and captures trending moves in both bull and bear markets.
Target: 15-25 trades/year per symbol by requiring confluence of trend, momentum, and regime.
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
    
    # Precompute session hours (08-20 UTC) once before loop
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # 1w data for trend context (loaded ONCE)
    df_1w = get_htf_data(prices, '1w')
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # 1d data for KAMA trend (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    # Calculate Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close_1d, n=10))  # |close[t] - close[t-10]|
    volatility = np.sum(np.abs(np.diff(close_1d, n=1)), axis=1)  # sum of |close[t] - close[t-1]| over 10 periods
    # Pad arrays to match length
    change = np.concatenate([np.full(10, np.nan), change])
    volatility = np.concatenate([np.full(10, np.nan), volatility])
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    # Calculate KAMA
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    # 1d data for RSI(14) (loaded ONCE)
    delta = np.diff(close_1d)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    # Pad first element
    gain = np.concatenate([[np.nan], gain])
    loss = np.concatenate([[np.nan], loss])
    # Wilder's smoothing
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[14] = np.nanmean(gain[1:15])  # first average
    avg_loss[14] = np.nanmean(loss[1:15])
    for i in range(15, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # 1d data for Choppiness Index(14) (loaded ONCE)
    # True Range
    tr1 = np.abs(np.diff(high))
    tr2 = np.abs(np.diff(low))
    tr3 = np.abs(high[1:] - low[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])  # align with close
    # Sum of TR over 14 periods
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    # Highest high and lowest low over 14 periods
    hh = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_sum / (hh - ll)) / np.log10(14)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for 1w EMA50 (50) and 1d indicators
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any data not ready
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or np.isnan(chop_aligned[i]) or
            np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        
        # Trend filters
        price_above_kama = curr_close > kama_aligned[i]
        price_below_kama = curr_close < kama_aligned[i]
        weekly_uptrend = curr_close > ema_50_1w_aligned[i]
        weekly_downtrend = curr_close < ema_50_1w_aligned[i]
        
        # Momentum confirmation
        rsi_long = rsi_aligned[i] > 50
        rsi_short = rsi_aligned[i] < 50
        
        # Regime filter: only trade in trending markets (chop < 50)
        trending_market = chop_aligned[i] < 50
        
        if position == 0:
            # Long entry: price > KAMA, RSI > 50, weekly uptrend, trending market
            if price_above_kama and rsi_long and weekly_uptrend and trending_market:
                signals[i] = 0.25
                position = 1
            # Short entry: price < KAMA, RSI < 50, weekly downtrend, trending market
            elif price_below_kama and rsi_short and weekly_downtrend and trending_market:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price < KAMA or RSI < 40 or chop > 60 (choppy)
            if curr_close < kama_aligned[i] or rsi_aligned[i] < 40 or chop_aligned[i] > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price > KAMA or RSI > 60 or chop > 60 (choppy)
            if curr_close > kama_aligned[i] or rsi_aligned[i] > 60 or chop_aligned[i] > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_KAMA_Trend_RSI_ChopFilter_v1"
timeframe = "1d"
leverage = 1.0