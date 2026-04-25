#!/usr/bin/env python3
"""
12h_KAMA_Trend_RSI_ChopFilter_v1
Hypothesis: Trade in direction of daily KAMA trend on 12h timeframe with RSI mean reversion and choppiness regime filter.
In trending markets (CHOP < 38.2): take pullbacks to RSI extremes in direction of daily KAMA.
In ranging markets (CHOP >= 38.2): fade RSI extremes at Bollinger Bands.
Position size: 0.25 to balance risk and reward.
Target: 12-30 trades/year to stay under 200 total trades on 12h.
Uses multiple timeframes: daily KAMA for trend, 12h RSI/BB/CHOP for entries.
Works in bull (buy pullbacks in uptrend) and bear (sell rallies in downtrend) markets.
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
    
    # Get 1d data for HTF KAMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d KAMA (adaptive moving average)
    close_1d = df_1d['close'].values
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close_1d, n=10))
    volatility = np.sum(np.abs(np.diff(close_1d, n=1)), axis=1)
    er = np.divide(change, volatility, out=np.zeros_like(change, dtype=float), where=volatility!=0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    # KAMA calculation
    kama = np.full_like(close_1d, np.nan, dtype=float)
    kama[29] = close_1d[29]  # seed
    for i in range(30, len(close_1d)):
        if not np.isnan(sc[i-10]) and not np.isnan(kama[i-1]):
            kama[i] = kama[i-1] + sc[i-10] * (close_1d[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    # Calculate 12h RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate 12h Bollinger Bands(20,2)
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + 2 * std_20
    lower_bb = sma_20 - 2 * std_20
    
    # Calculate 12h Choppiness Index(14)
    atr_14 = pd.Series(np.maximum(np.maximum(high - low, np.abs(high - np.roll(close, 1))), np.abs(low - np.roll(close, 1)))).rolling(window=14, min_periods=14).mean().values
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(np.sum(atr_14) / np.log(14) / (max_high - min_low))
    chop = np.where((max_high - min_low) == 0, 50, chop)  # avoid division by zero
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for all indicators
    start_idx = max(30, 20, 14)  # KAMA seed, BB, RSI/CHOP
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(upper_bb[i]) or np.isnan(lower_bb[i]) or np.isnan(chop[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine 1d KAMA trend (bullish = price above KAMA)
        htf_1d_bullish = close[i] > kama_aligned[i]
        htf_1d_bearish = close[i] < kama_aligned[i]
        
        if position == 0:
            # Long setup: RSI oversold + trend alignment + regime filter
            if htf_1d_bullish and chop[i] < 38.2:  # trending market
                long_setup = rsi[i] < 30  # pullback in uptrend
            else:  # ranging market
                long_setup = close[i] <= lower_bb[i] and rsi[i] < 30  # fade extreme
            
            # Short setup: RSI overbought + trend alignment + regime filter
            if htf_1d_bearish and chop[i] < 38.2:  # trending market
                short_setup = rsi[i] > 70  # rally in downtrend
            else:  # ranging market
                short_setup = close[i] >= upper_bb[i] and rsi[i] > 70  # fade extreme
            
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
            # Exit: RSI overbought OR trend reversal OR price touches upper BB in ranging
            if (rsi[i] > 70) or (not htf_1d_bullish and chop[i] >= 38.2) or (close[i] >= upper_bb[i] and chop[i] >= 38.2):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: RSI oversold OR trend reversal OR price touches lower BB in ranging
            if (rsi[i] < 30) or (not htf_1d_bearish and chop[i] >= 38.2) or (close[i] <= lower_bb[i] and chop[i] >= 38.2):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_KAMA_Trend_RSI_ChopFilter_v1"
timeframe = "12h"
leverage = 1.0