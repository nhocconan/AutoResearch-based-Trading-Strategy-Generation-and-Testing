#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1-day KAMA trend with RSI mean reversion and volume confirmation
# Long when 1-day KAMA is rising (bullish trend) AND RSI < 30 (oversold) AND volume > 1.5x average
# Short when 1-day KAMA is falling (bearish trend) AND RSI > 70 (overbought) AND volume > 1.5x average
# Uses daily KAMA for trend direction, RSI for mean-reversion entry, volume for confirmation
# Designed to work in bull markets via pullbacks in uptrends and in bear markets via bounces in downtrends
# Target: 20-30 trades per year (80-120 over 4 years) with 0.25 position sizing

name = "4h_1dKAMA_RSI_MeanReversion_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1-day KAMA ( Kaufman Adaptive Moving Average )
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Efficiency Ratio and Smoothing Constants for KAMA
    close_1d = df_1d['close'].values
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.sum(np.abs(np.diff(close_1d)), axis=0)  # Will compute properly below
    
    # Proper ER calculation
    er = np.zeros_like(close_1d)
    for i in range(1, len(close_1d)):
        if i >= 10:  # ER period
            direction = np.abs(close_1d[i] - close_1d[i-10])
            volatility = np.sum(np.abs(np.diff(close_1d[i-9:i+1])))
            er[i] = direction / volatility if volatility != 0 else 0
        else:
            er[i] = 0
    
    # Smoothing constants
    sc = (er * (2/2 - 2/30) + 2/30) ** 2  # fast=2, slow=30
    sc = np.where(er > 0, sc, 0)
    
    # KAMA calculation
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # KAMA trend: rising if current > previous, falling if current < previous
    kama_rising = kama > np.roll(kama, 1)
    kama_falling = kama < np.roll(kama, 1)
    kama_rising[0] = False
    kama_falling[0] = False
    
    # Align KAMA trend to 4h timeframe
    kama_rising_aligned = align_htf_to_ltf(prices, df_1d, kama_rising)
    kama_falling_aligned = align_htf_to_ltf(prices, df_1d, kama_falling)
    
    # RSI (14-period) on 4h close
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi[:13] = np.nan  # Not enough data for first 13 periods
    
    # Volume confirmation: >1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # Start after RSI/KAMA warmup
        # Skip if any critical value is NaN or outside session
        if (np.isnan(kama_rising_aligned[i]) or np.isnan(kama_falling_aligned[i]) or 
            np.isnan(rsi[i]) or np.isnan(volume_filter[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: KAMA rising (bullish trend) AND RSI oversold (<30) AND volume confirmation
            if kama_rising_aligned[i] and rsi[i] < 30 and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: KAMA falling (bearish trend) AND RSI overbought (>70) AND volume confirmation
            elif kama_falling_aligned[i] and rsi[i] > 70 and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: RSI overbought (>70) or KAMA turns bearish
            if rsi[i] > 70 or not kama_rising_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: RSI oversold (<30) or KAMA turns bullish
            if rsi[i] < 30 or not kama_falling_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals