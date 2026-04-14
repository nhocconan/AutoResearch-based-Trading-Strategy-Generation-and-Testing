#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using 1w KAMA for trend direction and 1d RSI(2) for mean reversion entries.
# In strong trends (KAMA slope > 0), buy RSI(2) < 10 oversold dips; in weak trends (KAMA slope < 0), sell RSI(2) > 90 overbought rallies.
# This combines trend following with mean reversion within the trend, working in both bull and bear markets.
# Volume confirmation filters low-quality signals. Target: 15-25 trades/year per symbol (60-100 total over 4 years).

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE for RSI(2)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate RSI(2) on 1d
    delta = np.diff(close_1d, prepend=np.nan)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/2, adjust=False, min_periods=2).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/2, adjust=False, min_periods=2).mean().values
    rs = avg_gain / avg_loss
    rs = np.where(avg_loss == 0, 100, rs)
    rsi_2 = 100 - (100 / (1 + rs))
    
    # Load 1w data ONCE for KAMA trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate Kaufman Adaptive Moving Average (KAMA)
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close_1w, k=10, prepend=np.full(10, np.nan)))
    volatility = np.nansum(np.abs(np.diff(close_1w, prepend=np.nan)), axis=0)
    # Fix: calculate rolling sum of volatility
    volatility_sum = np.zeros_like(volatility)
    for i in range(len(volatility_sum)):
        if i < 10:
            volatility_sum[i] = np.nan
        else:
            volatility_sum[i] = np.nansum(np.abs(np.diff(close_1w[max(0,i-9):i+1], prepend=np.nan)))
    er = np.where(volatility_sum != 0, change / volatility_sum, 0)
    
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    
    # Calculate KAMA
    kama = np.full_like(close_1w, np.nan)
    kama[0] = close_1w[0]
    for i in range(1, len(close_1w)):
        if np.isnan(sc[i]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc[i] * (close_1w[i] - kama[i-1])
    
    # KAMA slope: positive = uptrend, negative = downtrend
    kama_slope = np.diff(kama, prepend=np.nan)
    
    # Align indicators to lower timeframe
    rsi_2_aligned = align_htf_to_ltf(prices, df_1d, rsi_2)
    kama_slope_aligned = align_htf_to_ltf(prices, df_1w, kama_slope)
    
    # Volume confirmation: 1.5x average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(30, 20)  # Need KAMA and RSI
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(rsi_2_aligned[i]) or 
            np.isnan(kama_slope_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        volume_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # In uptrend (KAMA slope > 0), buy oversold dips (RSI < 10)
            # In downtrend (KAMA slope < 0), sell overbought rallies (RSI > 90)
            if (kama_slope_aligned[i] > 0 and 
                rsi_2_aligned[i] < 10 and 
                volume_confirmed):
                position = 1
                signals[i] = position_size
            elif (kama_slope_aligned[i] < 0 and 
                  rsi_2_aligned[i] > 90 and 
                  volume_confirmed):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: RSI returns to neutral (50) or trend changes
            if (rsi_2_aligned[i] >= 50 or 
                kama_slope_aligned[i] <= 0):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: RSI returns to neutral (50) or trend changes
            if (rsi_2_aligned[i] <= 50 or 
                kama_slope_aligned[i] >= 0):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_KAMATrend_RSI2MeanRev_v1"
timeframe = "1d"
leverage = 1.0