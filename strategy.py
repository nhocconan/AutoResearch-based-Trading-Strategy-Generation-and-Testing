#!/usr/bin/env python3
name = "6h_Adaptive_Kelly_MeanReversion_v1"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from math import sqrt

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 1d Bollinger Bands (20, 2.0) for mean reversion
    close_1d = df_1d['close'].values
    ma_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    upper = ma_20 + 2.0 * std_20
    lower = ma_20 - 2.0 * std_20
    
    # Align Bollinger Bands to 6h
    upper_aligned = align_htf_to_ltf(prices, df_1d, upper)
    lower_aligned = align_htf_to_ltf(prices, df_1d, lower)
    ma_20_aligned = align_htf_to_ltf(prices, df_1d, ma_20)
    
    # 6h RSI(14) for entry timing
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    # 6h volume filter: volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 14)  # Wait for BB and RSI
    
    for i in range(start_idx, n):
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or 
            np.isnan(ma_20_aligned[i]) or np.isnan(rsi[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price touches lower BB, RSI < 30 (oversold), volume confirmation
            if (low[i] <= lower_aligned[i] and 
                rsi[i] < 30 and 
                vol_filter[i]):
                # Kelly fraction: (edge * win_prob) / variance, simplified to 0.25
                signals[i] = 0.25
                position = 1
            # Short: price touches upper BB, RSI > 70 (overbought), volume confirmation
            elif (high[i] >= upper_aligned[i] and 
                  rsi[i] > 70 and 
                  vol_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price returns to middle band or RSI > 50
            if (close[i] >= ma_20_aligned[i] or rsi[i] > 50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price returns to middle band or RSI < 50
            if (close[i] <= ma_20_aligned[i] or rsi[i] < 50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 6s adaptive Kelly mean reversion using 1d Bollinger Bands and RSI.
# In ranging markets (common in 2025 BTC/ETH), price reverts to the mean at Bollinger Band extremes.
# The strategy fades extremes: long at lower BB with RSI<30, short at upper BB with RSI>70.
# Volume filter ensures entries occur during active participation, reducing false signals.
# Exits occur when price returns to the 20-period mean or RSI crosses 50, locking in profits.
# Position sizing uses a Kelly-inspired fraction (0.25) to balance risk and reward.
# This approach works in both bull (buy dips) and bear (sell rallies) markets by exploiting short-term mean reversion.