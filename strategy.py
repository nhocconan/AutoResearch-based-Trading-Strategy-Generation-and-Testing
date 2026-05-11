#!/usr/bin/env python3
name = "6h_RSI4_Reverse_WeeklyTrend_Filter"
timeframe = "6h"
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
    
    # Get weekly data for trend filter (primary)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Weekly RSI(4) for contrarian signal (extreme levels)
    close_1w = df_1w['close'].values
    delta = np.diff(close_1w, prepend=close_1w[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/4, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/4, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1w = 100 - (100 / (1 + rs))
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)
    
    # Get daily data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Daily volume spike: current volume > 1.5x 20-day average
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = vol_1d > (vol_ma_1d * 1.5)
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(rsi_1w_aligned[i]) or np.isnan(vol_spike_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Contrarian entry: extreme weekly RSI + volume spike
            # Long: weekly RSI < 15 (extreme oversold) + volume spike
            # Short: weekly RSI > 85 (extreme overbought) + volume spike
            if rsi_1w_aligned[i] < 15 and vol_spike_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            elif rsi_1w_aligned[i] > 85 and vol_spike_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: weekly RSI returns to neutral (>40) OR weekly trend changes
            if rsi_1w_aligned[i] > 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: weekly RSI returns to neutral (<60) OR weekly trend changes
            if rsi_1w_aligned[i] < 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals