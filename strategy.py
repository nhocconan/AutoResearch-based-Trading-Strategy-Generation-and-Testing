#!/usr/bin/env python3
# 1d_1w_engulfing_bounce_v1
# Hypothesis: Daily bullish/bearish engulfing candles with weekly EMA(21) trend filter and volume confirmation.
# Engulfing patterns signal strong reversals; weekly EMA ensures alignment with higher timeframe trend.
# Works in bull/bear markets as pattern is reversal-based and adapts to volatility.
# Target: 30-100 total trades over 4 years (7-25/year) to avoid fee drag.
# Uses engulfing detection + weekly trend + volume > 1.5x average.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_engulfing_bounce_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    open_price = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Weekly EMA(21) for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 21:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 21:
        ema_1w[20] = np.mean(close_1w[:21])
        for i in range(21, len(close_1w)):
            ema_1w[i] = (close_1w[i] * 2 + ema_1w[i-1] * 19) / 21
    
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Volume average (20-day)
    vol_ma_20 = np.full(n, np.nan)
    vol_sum = 0
    for i in range(n):
        vol_sum += volume[i]
        if i >= 20:
            vol_sum -= volume[i-20]
        if i >= 19:
            vol_ma_20[i] = vol_sum / 20
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data invalid
        if np.isnan(ema_1w_aligned[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        # Engulfing conditions
        bull_engulf = (close[i] > open_price[i-1]) and (open_price[i] < close[i-1]) and (close[i] - open_price[i] > close[i-1] - open_price[i-1])
        bear_engulf = (open_price[i] > close[i-1]) and (close[i] < open_price[i-1]) and (open_price[i] - close[i] > open_price[i-1] - close[i-1])
        
        # Volume confirmation
        vol_ok = volume[i] > vol_ma_20[i] * 1.5
        
        if position == 1:  # Long
            # Exit on bearish engulf or price below weekly EMA
            if bear_engulf or close[i] < ema_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short
            # Exit on bullish engulf or price above weekly EMA
            if bull_engulf or close[i] > ema_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long on bullish engulf with volume and above weekly EMA
            if bull_engulf and vol_ok and close[i] > ema_1w_aligned[i]:
                position = 1
                signals[i] = 0.25
            # Enter short on bearish engulf with volume and below weekly EMA
            elif bear_engulf and vol_ok and close[i] < ema_1w_aligned[i]:
                position = -1
                signals[i] = -0.25
    
    return signals