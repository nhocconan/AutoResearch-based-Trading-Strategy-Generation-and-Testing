#!/usr/bin/env python3
name = "6h_LiquiditySweep_1dTrend_Reversal"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_up_1d = close_1d > ema50_1d
    trend_up_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_up_1d)
    
    # Liquidity sweep detection: price breaches recent swing but reverses quickly
    # Swing high: highest high over 6 periods (36h lookback)
    swing_high = pd.Series(high).rolling(window=6, min_periods=6).max().shift(1).values
    # Swing low: lowest low over 6 periods (36h lookback)
    swing_low = pd.Series(low).rolling(window=6, min_periods=6).min().shift(1).values
    
    # Volume confirmation: current volume > 2x 6-period average (institutional participation)
    vol_ma6 = pd.Series(volume).rolling(window=6, min_periods=6).mean().values
    volume_spike = volume > 2.0 * vol_ma6
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for EMA and swing calculations
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(swing_high[i]) or np.isnan(swing_low[i]) or
            np.isnan(trend_up_1d_aligned[i]) or np.isnan(vol_ma6[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long setup: price sweeps below swing low (stop hunt) then reverses up
            # with volume spike and 1d uptrend intact
            if low[i] < swing_low[i] and close[i] > swing_low[i] and \
               trend_up_1d_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short setup: price sweeps above swing high (stop hunt) then reverses down
            # with volume spike and 1d downtrend
            elif high[i] > swing_high[i] and close[i] < swing_high[i] and \
                 not trend_up_1d_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks swing low (invalidation) OR 1d trend turns down
            if low[i] < swing_low[i] or not trend_up_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks swing high (invalidation) OR 1d trend turns up
            if high[i] > swing_high[i] or trend_up_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals