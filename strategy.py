#!/usr/bin/env python3
# 6h_Stochastic_RSI_Divergence_1wTrend
# Hypothesis: Stochastic RSI identifies overbought/oversold conditions with momentum.
# Divergences between price and Stochastic RSI signal potential reversals.
# Combined with weekly trend filter to trade in higher timeframe direction.
# Works in bull markets by taking long setups in uptrend, bear markets by taking short setups in downtrend.
# Target: 15-30 trades/year to minimize fee drag on 6h timeframe.

name = "6h_Stochastic_RSI_Divergence_1wTrend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate daily data for Stochastic RSI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate weekly EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Stochastic RSI on daily timeframe
    # Stochastic RSI = (RSI - min(RSI)) / (max(RSI) - min(RSI))
    rsi_period = 14
    stoch_period = 14
    k_period = 3
    d_period = 3
    
    # Calculate RSI
    delta = pd.Series(df_1d['close']).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/rsi_period, adjust=False, min_periods=rsi_period).mean()
    avg_loss = loss.ewm(alpha=1/rsi_period, adjust=False, min_periods=rsi_period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Calculate Stochastic RSI
    rsi_min = pd.Series(rsi_values).rolling(window=stoch_period, min_periods=stoch_period).min().values
    rsi_max = pd.Series(rsi_values).rolling(window=stoch_period, min_periods=stoch_period).max().values
    stoch_rsi = (rsi_values - rsi_min) / (rsi_max - rsi_min)
    stoch_rsi = np.where((rsi_max - rsi_min) == 0, 0.5, stoch_rsi)  # Avoid division by zero
    
    # Calculate %K and %D
    k_values = pd.Series(stoch_rsi).rolling(window=k_period, min_periods=k_period).mean().values
    d_values = pd.Series(k_values).rolling(window=d_period, min_periods=d_period).mean().values
    
    # Align weekly trend and Stochastic RSI to 6h timeframe
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    k_values_aligned = align_htf_to_ltf(prices, df_1d, k_values)
    d_values_aligned = align_htf_to_ltf(prices, df_1d, d_values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need weekly EMA50 (50), Stochastic RSI (14+14+3+3)
    start_idx = max(50, 14+14+3+3)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(k_values_aligned[i]) or 
            np.isnan(d_values_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Weekly trend filter
        uptrend = close[i] > ema_50_1w_aligned[i]
        downtrend = close[i] < ema_50_1w_aligned[i]
        
        if position == 0:
            # Long entry: oversold Stochastic RSI (%K crosses above %D) in uptrend
            if (uptrend and 
                k_values_aligned[i] > d_values_aligned[i] and 
                k_values_aligned[i-1] <= d_values_aligned[i-1] and
                k_values_aligned[i] < 0.3):  # Oversold threshold
                signals[i] = 0.25
                position = 1
            # Short entry: overbought Stochastic RSI (%K crosses below %D) in downtrend
            elif (downtrend and 
                  k_values_aligned[i] < d_values_aligned[i] and 
                  k_values_aligned[i-1] >= d_values_aligned[i-1] and
                  k_values_aligned[i] > 0.7):  # Overbought threshold
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: overbought condition or trend change
            if (k_values_aligned[i] >= 0.7 or not uptrend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: oversold condition or trend change
            if (k_values_aligned[i] <= 0.3 or not downtrend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals