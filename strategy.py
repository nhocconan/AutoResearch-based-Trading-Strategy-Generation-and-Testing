#!/usr/bin/env python3
# 4h_Keltner_Breakout_VolumeTrend
# Hypothesis: Keltner Channel breakout with volume confirmation and trend filter works in both bull and bear markets.
# In bull markets, buy breakouts above upper channel in uptrend; in bear markets, sell breakdowns below lower channel in downtrend.
# Volume confirms breakout strength. Target: 25-35 trades/year.

name = "4h_Keltner_Breakout_VolumeTrend"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Keltner Channel (20-period ATR, 1.5 multiplier)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    close_ma = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    kc_upper = close_ma + 1.5 * atr
    kc_lower = close_ma - 1.5 * atr
    
    # Volume confirmation: current volume > 1.5 * 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * vol_ma
    
    # 1-day trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_1d_up = close_1d > ema50_1d
    trend_1d_down = close_1d < ema50_1d
    
    # Align 1d trend to 4h
    trend_1d_up_aligned = align_htf_to_ltf(prices, df_1d, trend_1d_up.astype(float))
    trend_1d_down_aligned = align_htf_to_ltf(prices, df_1d, trend_1d_down.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Need enough data for all indicators
    
    for i in range(start_idx, n):
        if (np.isnan(kc_upper[i]) or np.isnan(kc_lower[i]) or
            np.isnan(volume_confirm[i]) or
            np.isnan(trend_1d_up_aligned[i]) or np.isnan(trend_1d_down_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: breakout above upper Keltner, volume confirmation, 1d uptrend
            if (close[i] > kc_upper[i] and
                volume_confirm[i] and
                trend_1d_up_aligned[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # Short: breakdown below lower Keltner, volume confirmation, 1d downtrend
            elif (close[i] < kc_lower[i] and
                  volume_confirm[i] and
                  trend_1d_down_aligned[i] > 0.5):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price crosses below middle line or 1d trend turns down
            if (close[i] < close_ma[i] or
                trend_1d_up_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price crosses above middle line or 1d trend turns up
            if (close[i] > close_ma[i] or
                trend_1d_down_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals