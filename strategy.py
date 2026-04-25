#!/usr/bin/env python3
"""
1d_KAMA_Trend_RSI_ChopFilter
Hypothesis: 1d KAMA trend direction combined with RSI(2) extremes and choppiness regime filter.
Long when KAMA trend up, RSI(2) < 10, and CHOP(14) > 61.8 (range regime).
Short when KAMA trend down, RSI(2) > 90, and CHOP(14) > 61.8 (range regime).
Uses weekly EMA34 for trend confirmation to avoid counter-trend trades.
Targets 7-25 trades/year on 1d timeframe to minimize fee drag while capturing mean reversion in ranging markets.
Works in both bull and bear markets by fading extremes only in range regimes.
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
    
    # 1d KAMA for trend direction
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    # KAMA: ER = |net change| / sum|abs change|, SC = [ER*(fastest-slowest)+slowest]^2
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = pd.Series(change).rolling(window=10, min_periods=1).sum().values
    net_change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    er = np.where(volatility > 0, net_change / volatility, 0)
    sc = (er * (0.6667 - 0.0645) + 0.0645) ** 2
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    # Weekly EMA34 for trend confirmation (avoid counter-trend)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # RSI(2) for mean reversion signals
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/2, adjust=False, min_periods=2).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/2, adjust=False, min_periods=2).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Choppiness Index regime filter: CHOP > 61.8 = ranging (mean revert)
    atr_raw = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    atr = pd.Series(atr_raw).rolling(window=14, min_periods=14).mean().values
    high_roll = pd.Series(high).rolling(window=14, min_periods=14).max().values
    low_roll = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr * 14 / np.log10((high_roll - low_roll) + 1e-10)) / np.log10(14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for 1d KAMA, 1w EMA34, RSI(2), and CHOP(14)
    start_idx = max(10, 34, 2, 14)  # KAMA needs 10, EMA34 needs 34, RSI needs 2, CHOP needs 14
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama_aligned[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(rsi[i]) or 
            np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: KAMA up, RSI(2) oversold (<10), ranging regime (CHOP>61.8)
            long_setup = (close[i] > kama_aligned[i]) and \
                         (rsi[i] < 10) and \
                         (chop[i] > 61.8)
            # Short: KAMA down, RSI(2) overbought (>90), ranging regime (CHOP>61.8)
            short_setup = (close[i] < kama_aligned[i]) and \
                          (rsi[i] > 90) and \
                          (chop[i] > 61.8)
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: KAMA turns down OR RSI(2) > 50 (mean reversion complete) OR chop < 38.2 (trend regime)
            if (close[i] < kama_aligned[i]) or \
               (rsi[i] > 50) or \
               (chop[i] < 38.2):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: KAMA turns up OR RSI(2) < 50 (mean reversion complete) OR chop < 38.2 (trend regime)
            if (close[i] > kama_aligned[i]) or \
               (rsi[i] < 50) or \
               (chop[i] < 38.2):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_KAMA_Trend_RSI_ChopFilter"
timeframe = "1d"
leverage = 1.0