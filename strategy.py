#!/usr/bin/env python3
# 1d_Weekly_RSI_Reversal_With_Volume_Confirmation
# Hypothesis: Uses weekly RSI extremes on Bitcoin and Ethereum to capture mean reversion opportunities.
# Enters long when weekly RSI < 30 (oversold) with volume confirmation on daily chart.
# Enters short when weekly RSI > 70 (overbought) with volume confirmation on daily chart.
# Designed for low trade frequency to avoid fee drag, with mean-reversion bias.
# Uses weekly timeframe for signal generation and daily for execution and confirmation.

name = "1d_Weekly_RSI_Reversal_With_Volume_Confirmation"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for RSI calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate weekly RSI(14)
    delta = np.diff(close_1w, prepend=close_1w[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_1w = 100 - (100 / (1 + rs))
    
    # Align weekly RSI to daily timeframe
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)
    
    # Daily volume confirmation: current volume > 1.5 * 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Ensure sufficient warmup for indicators
    
    for i in range(start_idx, n):
        if np.isnan(rsi_1w_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: weekly RSI < 30 (oversold) with volume confirmation
            if rsi_1w_aligned[i] < 30 and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: weekly RSI > 70 (overbought) with volume confirmation
            elif rsi_1w_aligned[i] > 70 and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: weekly RSI returns to neutral territory (>= 40)
            if rsi_1w_aligned[i] >= 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: weekly RSI returns to neutral territory (<= 60)
            if rsi_1w_aligned[i] <= 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals