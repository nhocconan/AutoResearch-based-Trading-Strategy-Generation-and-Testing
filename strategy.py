#!/usr/bin/env python3
# 1d_weekly_kama_rsi_chop_v2
# Hypothesis: Daily KAMA trend direction with RSI mean reversion and Choppiness Index regime filter.
# Long: KAMA rising (trend up), RSI < 30 (oversold), and CHOP > 61.8 (ranging market) → mean reversion long.
# Short: KAMA falling (trend down), RSI > 70 (overbought), and CHOP > 61.8 (ranging market) → mean reversion short.
# Exit: Opposing RSI extreme (RSI > 70 for long exit, RSI < 30 for short exit) or trend reversal.
# Uses 1d primary timeframe with 1w HTF for trend context (KAMA calculated on weekly close).
# Designed for low trade frequency (~10-25/year) to minimize fee drag in ranging/mean reverting markets.
# Works in bull markets via buying dips in uptrends and bear markets via selling rallies in downtrends.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_weekly_kama_rsi_chop_v2"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Weekly KAMA for trend direction
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Calculate Efficiency Ratio (ER) over 10 periods
    change_1w = np.abs(np.diff(close_1w, n=10))
    volatility_1w = np.sum(np.abs(np.diff(close_1w, n=1)), axis=0)[:len(change_1w)]
    er_1w = np.where(volatility_1w > 0, change_1w / volatility_1w, 0)
    # Smoothing constants
    sc_1w = (er_1w * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    # KAMA calculation
    kama_1w = np.full_like(close_1w, np.nan)
    kama_1w[29] = close_1w[29]  # seed
    for i in range(30, len(close_1w)):
        kama_1w[i] = kama_1w[i-1] + sc_1w[i-1] * (close_1w[i] - kama_1w[i-1])
    kama_1w_aligned = align_htf_to_ltf(prices, df_1w, kama_1w)
    
    # Daily RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Daily Choppiness Index(14)
    atr_1 = np.maximum(high[1:] - low[1:], np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])))
    atr_1 = np.concatenate([[np.nan], atr_1])
    atr_sum = pd.Series(atr_1).rolling(window=14, min_periods=14).sum().values
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_sum / (max_high - min_low)) / np.log10(14)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(kama_1w_aligned[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or
            np.isnan(close[i]) or np.isnan(high[i]) or np.isnan(low[i])):
            signals[i] = 0.0
            continue
        
        # KAMA trend: rising if current > previous, falling if current < previous
        kama_rising = kama_1w_aligned[i] > kama_1w_aligned[i-1]
        kama_falling = kama_1w_aligned[i] < kama_1w_aligned[i-1]
        
        # RSI extremes
        rsi_oversold = rsi[i] < 30
        rsi_overbought = rsi[i] > 70
        
        # Chop regime: ranging market
        chop_ranging = chop[i] > 61.8
        
        if position == 1:  # Long position
            # Exit: RSI overbought or trend turns down
            if rsi_overbought or not kama_rising:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: RSI oversold or trend turns up
            if rsi_oversold or not kama_falling:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Long entry: KAMA rising, RSI oversold, choppy market
            if kama_rising and rsi_oversold and chop_ranging:
                position = 1
                signals[i] = 0.25
            # Short entry: KAMA falling, RSI overbought, choppy market
            elif kama_falling and rsi_overbought and chop_ranging:
                position = -1
                signals[i] = -0.25
    
    return signals