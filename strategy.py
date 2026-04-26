#!/usr/bin/env python3
"""
4h_Keltner_Channel_Breakout_1dTrend_VolumeSpike
Hypothesis: 4h Keltner Channel (20,2.0) breakout with 1d EMA50 trend filter and volume confirmation (1.8x) to capture strong trending moves. Designed for low overtrading (~30-50 trades/year) by using volatility-based channels and tight volume filter. Works in bull/bear via 1d trend alignment.
"""

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
    
    # Load 1d data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    # Trend: 1 = uptrend (close > EMA50), -1 = downtrend (close < EMA50), 0 = invalid
    trend_1d = np.where(ema_50_1d_aligned > 0, 
                        np.where(close > ema_50_1d_aligned, 1, -1), 
                        0)
    
    # Calculate Keltner Channel (20, 2.0) on 4h
    # Typical Price = (high + low + close) / 3
    tp = (high + low + close) / 3.0
    # EMA of TP (20-period)
    ema_tp = pd.Series(tp).ewm(span=20, min_periods=20, adjust=False).mean().values
    # ATR (20-period)
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).ewm(span=20, min_periods=20, adjust=False).mean().values
    # Keltner Channel bands
    keltner_upper = ema_tp + (2.0 * atr)
    keltner_lower = ema_tp - (2.0 * atr)
    
    # Volume filter: volume > 1.8 * volume_ma(20) for confirmation
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 50 for EMA, 20 for TP EMA/ATR, 20 for volume MA)
    start_idx = max(50, 20, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(ema_tp[i]) or np.isnan(atr[i]) or
            np.isnan(keltner_upper[i]) or np.isnan(keltner_lower[i]) or np.isnan(volume_ma[i]) or
            np.isnan(trend_1d[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Keltner Channel breakout conditions
        if position == 0:
            # Long: Price breaks above Keltner Upper AND 1d uptrend AND volume spike (1.8x)
            if close[i] > keltner_upper[i] and trend_1d[i] == 1 and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Keltner Lower AND 1d downtrend AND volume spike (1.8x)
            elif close[i] < keltner_lower[i] and trend_1d[i] == -1 and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: Price falls below Keltner Lower OR 1d trend turns down
            if close[i] < keltner_lower[i] or trend_1d[i] == -1:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: Price rises above Keltner Upper OR 1d trend turns up
            if close[i] > keltner_upper[i] or trend_1d[i] == 1:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Keltner_Channel_Breakout_1dTrend_VolumeSpike"
timeframe = "4h"
leverage = 1.0