# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# 1d_KAMA_Trend_Follow_With_RSI_Confirmation
# Hypothesis: KAMA adapts to market noise, reducing whipsaws in choppy markets.
# In trending markets, price stays on one side of KAMA. RSI(14) filters
# overextended entries. Works in bull markets by following uptrends and
# bear markets by following downtrends. Uses 1w trend filter for higher timeframe
# confirmation to avoid counter-trend trades. Low trade frequency (~15-25/year)
# minimizes fee drag.

name = "1d_KAMA_Trend_Follow_With_RSI_Confirmation"
timeframe = "1d"
leverage = 1.0

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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly EMA20 for trend filter
    ema_20_1w = pd.Series(df_1w['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Calculate KAMA on daily close
    # Efficiency Ratio (ER) = |close - close[10]| / sum(|close - close[1]|) over 10 periods
    change = np.abs(close - np.roll(close, 10))
    volatility = np.sum(np.abs(np.diff(close, axis=0)), axis=0) if len(close) > 1 else 0
    # Vectorized volatility sum over 10 periods
    volatility_sum = np.zeros_like(close)
    for i in range(10, len(close)):
        volatility_sum[i] = np.sum(np.abs(close[i-9:i+1] - np.roll(close[i-9:i+1], 1)))
    er = np.zeros_like(close)
    er[10:] = change[10:] / np.where(volatility_sum[10:] == 0, 1, volatility_sum[10:])
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Calculate RSI(14)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(close)
    avg_loss = np.zeros_like(close)
    avg_gain[14] = np.mean(gain[1:15])
    avg_loss[14] = np.mean(loss[1:15])
    
    for i in range(15, len(close)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    
    # Align weekly EMA20 to daily
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need KAMA (10), RSI (14), weekly EMA (20)
    start_idx = max(20, 14)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(kama[i]) or 
            np.isnan(rsi[i]) or 
            np.isnan(ema_20_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Weekly trend filter
        weekly_uptrend = close[i] > ema_20_1w_aligned[i]
        weekly_downtrend = close[i] < ema_20_1w_aligned[i]
        
        # Price relative to KAMA
        price_above_kama = close[i] > kama[i]
        price_below_kama = close[i] < kama[i]
        
        # RSI filters for overextension
        rsi_not_overbought = rsi[i] < 70
        rsi_not_oversold = rsi[i] > 30
        
        if position == 0:
            # Long entry: weekly uptrend + price above KAMA + RSI not overbought
            if weekly_uptrend and price_above_kama and rsi_not_overbought:
                signals[i] = 0.25
                position = 1
            # Short entry: weekly downtrend + price below KAMA + RSI not oversold
            elif weekly_downtrend and price_below_kama and rsi_not_oversold:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: weekly trend turns down OR price crosses below KAMA
            if not weekly_uptrend or price_below_kama:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: weekly trend turns up OR price crosses above KAMA
            if not weekly_downtrend or price_above_kama:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals