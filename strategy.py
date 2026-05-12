#!/usr/bin/env python3
# 1d_1w_KAMA_Trend_RSI_Filter
# Hypothesis: On daily timeframe, use Kaufman Adaptive Moving Average (KAMA) for trend direction
# and RSI for momentum confirmation. Weekly timeframe provides higher timeframe trend filter
# to avoid counter-trend trades. Designed to work in both bull and bear markets by only
# taking trades aligned with the weekly trend. Low trade frequency expected due to dual timeframe
# confirmation and strict entry conditions.

name = "1d_1w_KAMA_Trend_RSI_Filter"
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
    volume = prices['volume'].values
    
    # === Daily KAMA (10, 2, 30) ===
    # Efficiency Ratio
    change = np.abs(np.diff(close, n=1))
    abs_change = np.abs(np.diff(close, n=1))
    # Pad first element
    change = np.insert(change, 0, 0)
    abs_change = np.insert(abs_change, 0, 0)
    
    # 10-period ER
    er_num = np.zeros(n)
    er_den = np.zeros(n)
    for i in range(10, n):
        er_num[i] = np.abs(close[i] - close[i-10])
        er_den[i] = np.sum(abs_change[i-9:i+1])
    
    er = np.divide(er_num, er_den, out=np.zeros_like(er_num), where=er_den!=0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    # KAMA
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # === Daily RSI (14) ===
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    for i in range(14, n):
        if i == 14:
            avg_gain[i] = np.mean(gain[1:15])
            avg_loss[i] = np.mean(loss[1:15])
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # === Weekly Trend Filter (EMA 34) ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or 
            np.isnan(ema_34_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Price relative to KAMA
        price_above_kama = close[i] > kama[i]
        price_below_kama = close[i] < kama[i]
        
        # RSI filters (avoid extremes, look for momentum)
        rsi_not_overbought = rsi[i] < 70
        rsi_not_oversold = rsi[i] > 30
        rsi_bullish = rsi[i] > 50
        rsi_bearish = rsi[i] < 50
        
        # Weekly trend filter
        weekly_uptrend = close[i] > ema_34_1w_aligned[i]
        weekly_downtrend = close[i] < ema_34_1w_aligned[i]
        
        if position == 0:
            # LONG: price above KAMA, RSI bullish but not overbought, weekly uptrend
            if price_above_kama and rsi_bullish and rsi_not_overbought and weekly_uptrend:
                signals[i] = 0.25
                position = 1
            # SHORT: price below KAMA, RSI bearish but not oversold, weekly downtrend
            elif price_below_kama and rsi_bearish and rsi_not_oversold and weekly_downtrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: price crosses below KAMA or RSI overbought
            if not price_above_kama or rsi[i] >= 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price crosses above KAMA or RSI oversold
            if not price_below_kama or rsi[i] <= 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals