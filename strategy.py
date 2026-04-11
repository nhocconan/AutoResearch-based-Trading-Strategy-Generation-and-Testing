#!/usr/bin/env python3
"""
1d_1w_kama_rsi_chop
Strategy: 1-day KAMA trend + RSI momentum + Choppiness regime filter
Timeframe: 1d
Leverage: 1.0
Hypothesis: Uses Kaufman Adaptive Moving Average (KAMA) to identify trend direction,
RSI for momentum confirmation, and Choppiness Index to filter ranging markets.
Designed for low trade frequency (<25/year) to minimize fee decay while capturing
trends in both bull and bear markets via adaptive trend strength.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_kama_rsi_chop"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load higher timeframe data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # === 1-day KAMA (trend direction) ===
    # Efficiency Ratio
    change = np.abs(np.diff(close, n=10))
    volatility = np.sum(np.abs(np.diff(close, n=1)), axis=0) if len(close) >= 11 else np.full(len(close)-10, np.nan)
    # Full array handling
    er = np.full_like(close, np.nan)
    if len(change) > 0 and len(volatility) > 0:
        er[10:] = change / np.where(volatility == 0, 1, volatility)
    # Smoothing constants
    sc = (er * 0.0645 + 0.0625) ** 2  # 2/(2+1) to 2/(30+1)
    # KAMA calculation
    kama = np.full_like(close, np.nan)
    if len(close) > 10:
        kama[10] = close[10]
        for i in range(11, len(close)):
            if not np.isnan(sc[i]):
                kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # === 1-day RSI (momentum) ===
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    rsi = np.concatenate([[np.nan] * 14, rsi])
    
    # === 1-day Choppiness Index (regime filter) ===
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Highest high and lowest low over 14 periods
    hh = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Chop calculation
    sum_atr = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    chop = 100 * np.log10(sum_atr / (hh - ll)) / np.log10(14)
    chop = np.concatenate([[np.nan] * 13, chop])  # align with 14-period lookback
    
    # === 1-week trend filter (optional bias) ===
    close_1w = df_1w['close'].values
    ema_8_1w = pd.Series(close_1w).ewm(span=8, adjust=False, min_periods=8).mean().values
    ema_21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_8_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_8_1w)
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # Session filter: 00-24 UTC (full day for daily timeframe)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 0) & (hours <= 23)  # always true for 1d, but keeps structure
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or
            np.isnan(ema_8_1w_aligned[i]) or np.isnan(ema_21_1w_aligned[i]) or
            not in_session[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        price_close = close[i]
        
        # Trend direction: price relative to KAMA
        above_kama = price_close > kama[i]
        below_kama = price_close < kama[i]
        
        # Momentum: RSI not extreme
        rsi_not_overbought = rsi[i] < 70
        rsi_not_oversold = rsi[i] > 30
        
        # Regime: trending market (Chop < 61.8)
        trending_market = chop[i] < 61.8
        
        # Weekly trend alignment (optional filter)
        weekly_uptrend = ema_8_1w_aligned[i] > ema_21_1w_aligned[i]
        weekly_downtrend = ema_8_1w_aligned[i] < ema_21_1w_aligned[i]
        
        # Long conditions: price above KAMA + RSI not overbought + trending market + weekly uptrend
        long_signal = above_kama and rsi_not_overbought and trending_market and weekly_uptrend
        
        # Short conditions: price below KAMA + RSI not oversold + trending market + weekly downtrend
        short_signal = below_kama and rsi_not_oversold and trending_market and weekly_downtrend
        
        # Exit when price crosses KAMA in opposite direction
        exit_long = position == 1 and price_close < kama[i]
        exit_short = position == -1 and price_close > kama[i]
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

# Hypothesis: Uses Kaufman Adaptive Moving Average (KAMA) to identify trend direction,
# RSI for momentum confirmation, and Choppiness Index to filter ranging markets.
# Designed for low trade frequency (<25/year) to minimize fee decay while capturing
# trends in both bull and bear markets via adaptive trend strength.