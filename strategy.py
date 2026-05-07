#!/usr/bin/env python3
name = "12h_1d_RSI_OverboughtOversold_Volume"
timeframe = "12h"
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
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 1d RSI (14)
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Use pandas for efficient calculation with proper min_periods
    gain_series = pd.Series(gain)
    loss_series = pd.Series(loss)
    avg_gain = gain_series.rolling(window=14, min_periods=14).mean().values
    avg_loss = loss_series.rolling(window=14, min_periods=14).mean().values
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi_1d = 100 - (100 / (1 + rs))
    
    # Align 1d RSI to 12h timeframe
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # 12h volume filter: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > 1.5 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Wait for indicators
    
    for i in range(start_idx, n):
        if np.isnan(rsi_1d_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: RSI < 30 (oversold) with volume confirmation
            if rsi_1d_aligned[i] < 30 and vol_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: RSI > 70 (overbought) with volume confirmation
            elif rsi_1d_aligned[i] > 70 and vol_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: RSI > 50 (mean reversion complete)
            if rsi_1d_aligned[i] > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: RSI < 50 (mean reversion complete)
            if rsi_1d_aligned[i] < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 12h timeframe captures medium-term swings while avoiding noise.
# 1d RSI identifies overbought (>70) and oversold (<30) conditions for mean reversion.
# Volume filter ensures trades occur with participation, reducing false signals.
# Works in both bull and bear markets by fading extremes.
# Target: 15-25 trades/year to minimize fee drag. Position size 0.25 limits risk.