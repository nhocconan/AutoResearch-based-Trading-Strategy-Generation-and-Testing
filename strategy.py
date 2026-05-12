#!/usr/bin/env python3
# 6h_StochasticRSI_1dTrend_Volume
# Hypothesis: Use Stochastic RSI (14,14,3,3) to identify oversold/overbought conditions on 6h,
# filtered by 1d EMA50 for trend direction and volume confirmation. Designed for low frequency
# (15-30 trades/year) with clear entry/exit rules to avoid whipsaws in both bull and bear markets.
# Stochastic RSI helps identify turning points while respecting higher timeframe trend.

name = "6h_StochasticRSI_1dTrend_Volume"
timeframe = "6h"
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
    
    # === 1d EMA50 for trend filter ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === Stochastic RSI on 6h (14,14,3,3) ===
    rsi_period = 14
    stoch_period = 14
    k_period = 3
    d_period = 3
    
    # Calculate RSI
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/rsi_period, adjust=False, min_periods=rsi_period).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/rsi_period, adjust=False, min_periods=rsi_period).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate Stochastic RSI
    rsi_min = pd.Series(rsi).rolling(window=stoch_period, min_periods=stoch_period).min().values
    rsi_max = pd.Series(rsi).rolling(window=stoch_period, min_periods=stoch_period).max().values
    stoch_rsi = (rsi - rsi_min) / (rsi_max - rsi_min + 1e-10) * 100
    
    # Calculate %K and %D
    k = pd.Series(stoch_rsi).rolling(window=k_period, min_periods=k_period).mean().values
    d = pd.Series(k).rolling(window=d_period, min_periods=d_period).mean().values
    
    # === Volume confirmation (24-period average) ===
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(rsi_period + stoch_period + k_period + d_period, 50) + 24
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(k[i]) or np.isnan(d[i]) or 
            np.isnan(vol_ma_24[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 1d EMA50
        trend_up = close[i] > ema_50_1d_aligned[i]
        trend_down = close[i] < ema_50_1d_aligned[i]
        
        # Stochastic RSI conditions
        oversold = k[i] < 20 and d[i] < 20
        overbought = k[i] > 80 and d[i] > 80
        
        # Volume filter: above average
        vol_ok = volume[i] > vol_ma_24[i]
        
        if position == 0:
            # LONG: oversold reversal in uptrend with volume
            if oversold and trend_up and vol_ok:
                signals[i] = 0.25
                position = 1
            # SHORT: overbought reversal in downtrend with volume
            elif overbought and trend_down and vol_ok:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: overbought condition or trend reversal
            if overbought or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: oversold condition or trend reversal
            if oversold or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals