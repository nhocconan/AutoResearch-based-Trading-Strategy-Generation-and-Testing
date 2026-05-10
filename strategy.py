#!/usr/bin/env python3
"""
4h_Contrarian_Trend_Reversal
Hypothesis: Counter-trend reversals at 1d Bollinger Bands with 4h momentum confirmation.
Works in bull/bear: Fades extremes in ranging markets, follows momentum in trends.
Targets 25-40 trades/year by requiring Bollinger touch + momentum divergence + volume.
"""

name = "4h_Contrarian_Trend_Reversal"
timeframe = "4h"
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
    
    # Get 1d data for Bollinger Bands and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d Bollinger Bands (20, 2)
    close_1d = pd.Series(df_1d['close'])
    bb_middle = close_1d.rolling(window=20, min_periods=20).mean().values
    bb_std = close_1d.rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    
    bb_upper_aligned = align_htf_to_ltf(prices, df_1d, bb_upper)
    bb_lower_aligned = align_htf_to_ltf(prices, df_1d, bb_lower)
    
    # Calculate 1d trend filter (EMA50)
    ema_50_1d = close_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 4h RSI(14) for momentum
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate 4h volume average (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need 1d BB (20), 1d EMA50 (50), 4h RSI (14), 4h vol avg (20)
    start_idx = max(50, 20, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(bb_upper_aligned[i]) or 
            np.isnan(bb_lower_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(rsi[i]) or 
            np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # 1d trend filter
        uptrend_1d = close[i] > ema_50_1d_aligned[i]
        downtrend_1d = close[i] < ema_50_1d_aligned[i]
        
        # Volume filter: current volume > 1.3x average
        volume_filter = volume[i] > vol_avg[i] * 1.3
        
        if position == 0:
            # Long setup: price touches/below lower BB + oversold RSI + volume
            if close[i] <= bb_lower_aligned[i] and rsi[i] < 30 and volume_filter:
                # In uptrend, wait for pullback; in downtrend, fade the extreme
                if uptrend_1d or (downtrend_1d and rsi[i] < 25):
                    signals[i] = 0.25
                    position = 1
            # Short setup: price touches/above upper BB + overbought RSI + volume
            elif close[i] >= bb_upper_aligned[i] and rsi[i] > 70 and volume_filter:
                # In downtrend, wait for bounce; in uptrend, fade the extreme
                if downtrend_1d or (uptrend_1d and rsi[i] > 75):
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: RSI reaches midline or price hits upper BB
            if rsi[i] > 50 or close[i] >= bb_upper_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: RSI reaches midline or price hits lower BB
            if rsi[i] < 50 or close[i] <= bb_lower_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals