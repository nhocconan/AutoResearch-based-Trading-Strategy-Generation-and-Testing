#!/usr/bin/env python3
# 12h_KAMA_Trend_With_RSI_Filter_and_Volume
# Hypothesis: On 12h timeframe, KAMA (Kaufman Adaptive Moving Average) captures trend direction,
# RSI (14) filters overbought/oversold conditions, and volume surge confirms institutional participation.
# Works in bull/bear: KAMA adapts to market noise, RSI prevents chasing extremes, volume ensures validity.
# Uses 1-week EMA200 for higher timeframe trend filter to avoid counter-trend trades in strong trends.

name = "12h_KAMA_Trend_With_RSI_Filter_and_Volume"
timeframe = "12h"
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
    
    # KAMA parameters
    er_length = 10
    fast_sc = 2 / (2 + 1)  # EMA(2)
    slow_sc = 2 / (30 + 1) # EMA(30)
    
    # Calculate Efficiency Ratio (ER)
    change = np.abs(np.diff(close, n=er_length))
    volatility = np.sum(np.abs(np.diff(close)), axis=0)  # placeholder, will compute properly below
    
    # Proper ER calculation
    er = np.full(n, np.nan)
    for i in range(er_length, n):
        price_change = np.abs(close[i] - close[i-er_length])
        price_volatility = np.sum(np.abs(np.diff(close[i-er_length+1:i+1])))
        if price_volatility > 0:
            er[i] = price_change / price_volatility
        else:
            er[i] = 0
    
    # Smoothing constant
    sc = np.full(n, np.nan)
    for i in range(er_length, n):
        sc[i] = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama = np.full(n, np.nan)
    kama[er_length] = close[er_length]  # seed
    for i in range(er_length + 1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI (14)
    rsi_period = 14
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)
    rsi = np.full(n, np.nan)
    
    if n >= rsi_period + 1:
        avg_gain[rsi_period] = np.mean(gain[1:rsi_period+1])
        avg_loss[rsi_period] = np.mean(loss[1:rsi_period+1])
        for i in range(rsi_period + 1, n):
            avg_gain[i] = (avg_gain[i-1] * (rsi_period-1) + gain[i]) / rsi_period
            avg_loss[i] = (avg_loss[i-1] * (rsi_period-1) + loss[i]) / rsi_period
            rs = avg_gain[i] / avg_loss[i] if avg_loss[i] != 0 else 0
            rsi[i] = 100 - (100 / (1 + rs))
    
    # Volume ratio (current / 20-period average)
    vol_ma = np.full(n, np.nan)
    if n >= 20:
        vol_ma[19] = np.mean(volume[0:20])
        for i in range(20, n):
            vol_ma[i] = (vol_ma[i-1] * 19 + volume[i]) / 20
    
    volume_ratio = np.full(n, np.nan)
    valid_vol = (~np.isnan(vol_ma)) & (vol_ma != 0)
    volume_ratio[valid_vol] = volume[valid_vol] / vol_ma[valid_vol]
    
    # 1-week EMA200 for trend filter (higher timeframe)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_200_1w = np.full_like(close_1w, np.nan)
    if len(close_1w) >= 200:
        ema_200_1w[199] = np.mean(close_1w[0:200])
        for i in range(200, len(close_1w)):
            ema_200_1w[i] = (ema_200_1w[i-1] * 199 + close_1w[i]) / 200
    
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(er_length, rsi_period + 1, 20)  # Ensure all indicators are ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(volume_ratio[i]) or
            np.isnan(ema_200_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price > KAMA (uptrend), RSI < 70 (not overbought), volume surge, and above weekly EMA200
            if (close[i] > kama[i] and 
                rsi[i] < 70 and 
                volume_ratio[i] > 2.0 and
                close[i] > ema_200_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price < KAMA (downtrend), RSI > 30 (not oversold), volume surge, and below weekly EMA200
            elif (close[i] < kama[i] and 
                  rsi[i] > 30 and 
                  volume_ratio[i] > 2.0 and
                  close[i] < ema_200_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price < KAMA OR RSI > 80 (extreme overbought)
            if close[i] < kama[i] or rsi[i] > 80:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price > KAMA OR RSI < 20 (extreme oversold)
            if close[i] > kama[i] or rsi[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals