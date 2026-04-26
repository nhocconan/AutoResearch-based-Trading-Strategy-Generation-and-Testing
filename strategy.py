#!/usr/bin/env python3
"""
1d_KAMA_Trend_RSI_ChopFilter
Hypothesis: On daily timeframe, enter long when KAMA trend is up (price > KAMA) and RSI < 30 in choppy market (CHOP > 61.8), enter short when KAMA trend is down (price < KAMA) and RSI > 70 in choppy market. Uses 1w HTF for regime filter: only trade when 1d price is above/below 1w EMA50 to align with weekly trend. Designed for 7-25 trades/year (30-100 total over 4 years) to avoid fee drag. Works in both bull and bear markets by combining trend-following (KAMA) with mean-reversion (RSI extremes) in choppy regimes, filtered by weekly trend alignment.
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
    
    # Load 1d data ONCE before loop for KAMA, RSI, CHOP
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Load 1w data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # 1d KAMA (ER=10, fast=2, slow=30)
    close_1d = df_1d['close'].values
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.sum(np.abs(np.diff(close_1d, prepend=close_1d[0])), axis=0) if False else None  # placeholder
    # Correct ER calculation
    er = np.zeros_like(close_1d)
    for i in range(1, len(close_1d)):
        if i < 10:
            er[i] = 1.0  # not enough data, use default
        else:
            direction = np.abs(close_1d[i] - close_1d[i-10])
            volatility = np.sum(np.abs(np.diff(close_1d[i-9:i+1])))
            er[i] = direction / volatility if volatility != 0 else 0
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    # 1d RSI(14)
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / np.where(avg_loss == 0, 1e-10, avg_loss)
    rsi = 100 - (100 / (1 + rs))
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # 1d CHOPPINESS INDEX(14)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close_1d, 1))
    tr3 = np.abs(low - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    chop = 100 * np.log10(tr_sum / (atr * 14)) / np.log10(14)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25  # Position size
    
    # Warmup: max of calculations (14 for RSI/CHOP, 10 for KAMA ER, 50 for EMA)
    start_idx = max(14, 10, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama_aligned[i]) or 
            np.isnan(rsi_aligned[i]) or 
            np.isnan(chop_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i])):
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        close_val = close[i]
        kama_val = kama_aligned[i]
        rsi_val = rsi_aligned[i]
        chop_val = chop_aligned[i]
        ema_50_1w_val = ema_50_1w_aligned[i]
        
        # Determine 1d trend vs KAMA: bullish if price > KAMA, bearish if price < KAMA
        bullish_kama = close_val > kama_val
        bearish_kama = close_val < kama_val
        
        # Choppy market regime: CHOP > 61.8 = ranging (mean revert)
        choppy = chop_val > 61.8
        
        # Weekly trend filter: only trade in direction of weekly EMA50
        weekly_uptrend = close_val > ema_50_1w_val
        weekly_downtrend = close_val < ema_50_1w_val
        
        # Entry conditions: RSI extremes in choppy market, aligned with weekly trend
        long_entry = bullish_kama and rsi_val < 30 and choppy and weekly_uptrend
        short_entry = bearish_kama and rsi_val > 70 and choppy and weekly_downtrend
        
        # Exit conditions: opposite RSI extreme or loss of choppy regime
        exit_long = rsi_val > 70 or not choppy
        exit_short = rsi_val < 30 or not choppy
        
        if long_entry and position != 1:
            signals[i] = base_size
            position = 1
        elif short_entry and position != -1:
            signals[i] = -base_size
            position = -1
        elif position == 1 and exit_long:
            signals[i] = 0.0
            position = 0
        elif position == -1 and exit_short:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
    
    return signals

name = "1d_KAMA_Trend_RSI_ChopFilter"
timeframe = "1d"
leverage = 1.0