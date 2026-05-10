#!/usr/bin/env python3
# 1d_KAMA_RSI_ChopFilter
# Hypothesis: On 1d timeframe, KAMA (adaptive trend) identifies market direction,
# RSI(2) identifies overextended conditions for mean reversion, and Choppiness Index
# filters for ranging markets (CHOP > 61.8) where mean reversion works best.
# In trending markets (CHOP < 38.2), we follow KAMA direction with momentum.
# This combines trend-following and mean-reversion with regime filtering to work
# in both bull and bear markets while keeping trade frequency low (target: 10-30/year).

name = "1d_KAMA_RSI_ChopFilter"
timeframe = "1d"
leverage = 1.0

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
    
    # Get weekly data for trend filter (1w)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate KAMA on daily close
    # KAMA parameters: ER period=10, FAST=2, SLOW=30
    close_series = pd.Series(close)
    change = abs(close_series.diff(1))
    volatility = change.rolling(window=10, min_periods=10).sum()
    ER = change / volatility.replace(0, np.nan)
    ER = ER.fillna(0)
    SC = (ER * (2/2 - 2/30) + 2/30) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + SC[i] * (close[i] - kama[i-1])
    
    # Calculate RSI(2) for mean reversion signals
    delta = close_series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=2, min_periods=2).mean()
    avg_loss = loss.rolling(window=2, min_periods=2).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50)  # neutral when undefined
    
    # Calculate Choppiness Index (14-period)
    # CHOP = 100 * log10(sum(ATR(1)) / (max(high) - min(low))) / log10(14)
    atr1 = np.maximum(high - low, np.maximum(abs(high - np.roll(close, 1)), abs(low - np.roll(close, 1))))
    atr1[0] = high[0] - low[0]  # first ATR
    atr_sum = pd.Series(atr1).rolling(window=14, min_periods=14).sum()
    roll_max = pd.Series(high).rolling(window=14, min_periods=14).max()
    roll_min = pd.Series(low).rolling(window=14, min_periods=14).min()
    chop = 100 * np.log10(atr_sum / (roll_max - roll_min)) / np.log10(14)
    chop = chop.fillna(50)  # neutral when undefined
    
    # Align weekly trend (using weekly close EMA20)
    ema_20_1w = pd.Series(df_1w['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need KAMA (need ~10), RSI(2) (2), CHOP (14), weekly EMA (20)
    start_idx = max(20, 14)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(kama[i]) or 
            np.isnan(rsi[i]) or 
            np.isnan(chop[i]) or 
            np.isnan(ema_20_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Regime filter: Choppiness Index
        ranging = chop[i] > 61.8  # mean reversion regime
        trending = chop[i] < 38.2  # trend following regime
        
        # Weekly trend filter
        weekly_uptrend = close[i] > ema_20_1w_aligned[i]
        weekly_downtrend = close[i] < ema_20_1w_aligned[i]
        
        if position == 0:
            # Long entry conditions
            if ranging and rsi[i] < 30 and close[i] > kama[i]:
                # Oversold in ranging market, price above KAMA (bullish bias)
                signals[i] = 0.25
                position = 1
            elif trending and weekly_uptrend and close[i] > kama[i]:
                # Uptrend and price above KAMA
                signals[i] = 0.25
                position = 1
            # Short entry conditions
            elif ranging and rsi[i] > 70 and close[i] < kama[i]:
                # Overbought in ranging market, price below KAMA (bearish bias)
                signals[i] = -0.25
                position = -1
            elif trending and weekly_downtrend and close[i] < kama[i]:
                # Downtrend and price below KAMA
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: opposite RSI extreme or trend change
            if (ranging and rsi[i] > 70) or (trending and not weekly_uptrend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: opposite RSI extreme or trend change
            if (ranging and rsi[i] < 30) or (trending and not weekly_downtrend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals