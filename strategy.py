#!/usr/bin/env python3
"""
1d_KAMA_RSI_Chop
Hypothesis: Kaufman Adaptive Moving Average (KAMA) provides adaptive trend direction,
Relative Strength Index (RSI) identifies overbought/oversold conditions,
and Choppiness Index (CHOP) filters ranging vs trending markets.
Long when KAMA rising, RSI < 50, and CHOP > 61.8 (ranging market for mean reversion).
Short when KAMA falling, RSI > 50, and CHOP > 61.8.
Designed for low trade frequency (<25/year) to avoid fee drag in choppy markets.
"""

name = "1d_KAMA_RSI_Chop"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # KAMA parameters
    kama_period = 10
    fast_ema = 2
    slow_ema = 30
    
    # Calculate Efficiency Ratio (ER)
    change = np.abs(close - np.roll(close, kama_period))
    change[0:kama_period] = 0
    
    volatility = np.sum(np.abs(np.diff(close)), axis=0)  # placeholder, will compute properly below
    # Proper volatility calculation
    volatility = np.zeros_like(close)
    for i in range(1, len(close)):
        volatility[i] = volatility[i-1] + np.abs(close[i] - close[i-1])
    
    # Avoid division by zero
    er = np.zeros_like(close)
    for i in range(kama_period, len(close)):
        if volatility[i] != 0:
            er[i] = change[i] / volatility[i]
        else:
            er[i] = 0
    
    # Smoothing constants
    sc = (er * (2/(fast_ema+1) - 2/(slow_ema+1)) + 2/(slow_ema+1)) ** 2
    
    # Calculate KAMA
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI (14)
    rsi_period = 14
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(close)
    avg_loss = np.zeros_like(close)
    avg_gain[rsi_period] = np.mean(gain[1:rsi_period+1])
    avg_loss[rsi_period] = np.mean(loss[1:rsi_period+1])
    
    for i in range(rsi_period+1, len(close)):
        avg_gain[i] = (avg_gain[i-1] * (rsi_period-1) + gain[i]) / rsi_period
        avg_loss[i] = (avg_loss[i-1] * (rsi_period-1) + loss[i]) / rsi_period
    
    rs = np.zeros_like(close)
    rs[avg_loss != 0] = avg_gain[avg_loss != 0] / avg_loss[avg_loss != 0]
    rsi = np.zeros_like(close)
    rsi = 100 - (100 / (1 + rs))
    rsi[avg_loss == 0] = 100
    
    # Choppiness Index (14)
    chop_period = 14
    atr = np.zeros_like(close)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR calculation
    atr[chop_period-1] = np.mean(tr[:chop_period])
    for i in range(chop_period, len(close)):
        atr[i] = (atr[i-1] * (chop_period-1) + tr[i]) / chop_period
    
    # Choppiness Index
    sum_atr = np.zeros_like(close)
    for i in range(chop_period-1, len(close)):
        if i == chop_period-1:
            sum_atr[i] = np.sum(atr[i-chop_period+1:i+1])
        else:
            sum_atr[i] = sum_atr[i-1] - atr[i-chop_period] + atr[i]
    
    max_high = np.zeros_like(close)
    min_low = np.zeros_like(close)
    for i in range(chop_period-1, len(close)):
        max_high[i] = np.max(high[i-chop_period+1:i+1])
        min_low[i] = np.min(low[i-chop_period+1:i+1])
    
    chop = np.zeros_like(close)
    for i in range(chop_period-1, len(close)):
        if max_high[i] - min_low[i] != 0:
            chop[i] = 100 * np.log10(sum_atr[i] / (max_high[i] - min_low[i])) / np.log10(chop_period)
        else:
            chop[i] = 50
    
    # Weekly trend filter (1w)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) > 0:
        weekly_close = df_1w['close'].values
        weekly_ema = pd.Series(weekly_close).ewm(span=20, adjust=False, min_periods=20).mean().values
        weekly_ema_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema)
    else:
        weekly_ema_aligned = np.full(n, np.nan)
    
    # Signals
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if weekly EMA not available
        if np.isnan(weekly_ema_aligned[i]):
            signals[i] = 0.0
            continue
            
        weekly_uptrend = close[i] > weekly_ema_aligned[i]
        weekly_downtrend = close[i] < weekly_ema_aligned[i]
        
        if position == 0:
            # LONG: KAMA rising, RSI < 50, Chop > 61.8 (ranging market), weekly uptrend
            if kama[i] > kama[i-1] and rsi[i] < 50 and chop[i] > 61.8 and weekly_uptrend:
                signals[i] = 0.25
                position = 1
            # SHORT: KAMA falling, RSI > 50, Chop > 61.8 (ranging market), weekly downtrend
            elif kama[i] < kama[i-1] and rsi[i] > 50 and chop[i] > 61.8 and weekly_downtrend:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: KAMA falling or RSI > 60 or Chop < 38.2 (trending) or weekly trend fails
            if kama[i] < kama[i-1] or rsi[i] > 60 or chop[i] < 38.2 or not weekly_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: KAMA rising or RSI < 40 or Chop < 38.2 (trending) or weekly trend fails
            if kama[i] > kama[i-1] or rsi[i] < 40 or chop[i] < 38.2 or not weekly_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals