#!/usr/bin/env python3
"""
4h_VolatilityBreakout_1dTrend_Confirmation_v1
Hypothesis: Buy when price breaks above Bollinger Upper Band with volume surge in an uptrend (price > EMA50), sell when breaks below BB Lower Band with volume surge in a downtrend (price < EMA50). Uses 1d EMA50 for trend filter to avoid counter-trend trades. Bollinger Bands capture volatility expansion, which often precedes strong moves. Volume surge confirms institutional participation. Designed for 4h timeframe with ~20-40 trades/year to minimize fee drag.
"""

name = "4h_VolatilityBreakout_1dTrend_Confirmation_v1"
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
    
    # === 1D Data for Trend Filter ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Trend filter: EMA50 on 1d close
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Bollinger Bands on 4h close (20, 2)
    bb_period = 20
    bb_std = 2
    close_series = pd.Series(close)
    bb_ma = close_series.rolling(window=bb_period, min_periods=bb_period).mean().values
    bb_std_dev = close_series.rolling(window=bb_period, min_periods=bb_period).std().values
    bb_upper = bb_ma + bb_std_dev * bb_std
    bb_lower = bb_ma - bb_std_dev * bb_std
    
    # Volume spike: current volume > 2.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = bb_period
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_50_aligned[i]) or 
            np.isnan(bb_upper[i]) or 
            np.isnan(bb_lower[i]) or 
            np.isnan(bb_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above BB Upper AND uptrend (price > EMA50) AND volume spike
            if close[i] > bb_upper[i] and close[i] > ema_50_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below BB Lower AND downtrend (price < EMA50) AND volume spike
            elif close[i] < bb_lower[i] and close[i] < ema_50_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below BB Middle OR reverses below EMA50
            if close[i] < bb_ma[i] or close[i] < ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: price crosses above BB Middle OR reverses above EMA50
            if close[i] > bb_ma[i] or close[i] > ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals