#!/usr/bin/env python3
"""
4h_Camarilla_R3_S3_Breakout_1wEMA50_Trend_VolumeSpike
Hypothesis: Camarilla R3/S3 breakout on 4h with 1-week EMA50 trend filter and volume confirmation (>2.0x 20-period MA).
Long when price breaks above R3 with uptrend and volume spike.
Short when price breaks below S3 with downtrend and volume spike.
Uses discrete position sizing (0.30) to balance return and drawdown.
Designed for low trade frequency (<400 total 4h trades) to minimize fee drag and improve test generalization.
Works in both bull and bear markets by following the 1-week trend, which adapts to regime changes.
Target: 20-50 trades/year (80-200 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:  # warmup for 1w EMA50 + 4h indicators
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # 1w EMA50 trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    uptrend_1w = close > ema_50_1w_aligned
    downtrend_1w = close < ema_50_1w_aligned
    
    # Get 1d data for Camarilla levels (pivots from prior 1d)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: R3, S3 based on prior 1d OHLC
    # R3 = close + 1.1*(high - low)
    # S3 = close - 1.1*(high - low)
    R3 = close_1d + 1.1 * (high_1d - low_1d)
    S3 = close_1d - 1.1 * (high_1d - low_1d)
    
    # Align Camarilla levels to 4h (wait for 1d close)
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    
    # Volume confirmation: volume > 2.0x 20-period MA (strict to reduce trades)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 50 for 1w EMA + 2 for 1d + 20 for volume MA)
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(R3_aligned[i]) or 
            np.isnan(S3_aligned[i]) or np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.30
            else:
                signals[i] = -0.30
            continue
        
        if position == 0:
            # Long: price breaks above R3 with 1w uptrend and volume spike
            if (close[i] > R3_aligned[i] and uptrend_1w[i] and volume_spike[i]):
                signals[i] = 0.30
                position = 1
            # Short: price breaks below S3 with 1w downtrend and volume spike
            elif (close[i] < S3_aligned[i] and downtrend_1w[i] and volume_spike[i]):
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.30
            # Exit: 1w trend changes to downtrend OR price re-enters Camarilla H3-L3 (mean reversion)
            H3 = close_1d[i-1] + 1.1/2 * (high_1d[i-1] - low_1d[i-1]) if i-1 >= 0 else R3_aligned[i]
            L3 = close_1d[i-1] - 1.1/2 * (high_1d[i-1] - low_1d[i-1]) if i-1 >= 0 else S3_aligned[i]
            H3_aligned = align_htf_to_ltf(prices, df_1d, np.full_like(close, H3))[i] if i-1 >= 0 else R3_aligned[i]
            L3_aligned = align_htf_to_ltf(prices, df_1d, np.full_like(close, L3))[i] if i-1 >= 0 else S3_aligned[i]
            if (not uptrend_1w[i] or (close[i] < H3_aligned[i] and close[i] > L3_aligned[i])):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.30
            # Exit: 1w trend changes to uptrend OR price re-enters Camarilla H3-L3
            if (not downtrend_1w[i] or (close[i] < H3_aligned[i] and close[i] > L3_aligned[i])):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R3_S3_Breakout_1wEMA50_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0