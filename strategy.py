#!/usr/bin/env python3
# 4h_Donchian20_With_1dTrend_Volume
# Hypothesis: Donchian channel breakouts capture strong directional moves.
# Combined with 1d EMA trend filter and volume confirmation to reduce whipsaw.
# Designed for low turnover (~20-30 trades/year) to minimize fee drag in both bull and bear markets.
# Exit on opposite Donchian band touch or trend reversal.
# Target: < 100 total trades over 4 years.

name = "4h_Donchian20_With_1dTrend_Volume"
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
    
    # === Donchian Channel (20-period) ===
    lookback = 20
    donchian_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    donchian_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # === 1d EMA50 Trend Filter ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema50_1d_4h = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # === Volume Spike Filter (20-period EMA) ===
    vol_ema20 = pd.Series(volume).ewm(span=20, min_periods=20, adjust=False).mean().values
    volume_ok = volume > vol_ema20 * 1.8  # Require 1.8x average volume
    
    # === Signal Parameters ===
    position_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for Donchian and EMA)
    start_idx = max(lookback, 50)
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema50_1d_4h[i]) or np.isnan(volume_ok[i])):
            if position == 1:
                signals[i] = 0.0
            elif position == -1:
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Break above Donchian high + above 1d EMA50 + volume spike
            if close[i] > donchian_high[i] and close[i] > ema50_1d_4h[i] and volume_ok[i]:
                signals[i] = position_size
                position = 1
            # Short: Break below Donchian low + below 1d EMA50 + volume spike
            elif close[i] < donchian_low[i] and close[i] < ema50_1d_4h[i] and volume_ok[i]:
                signals[i] = -position_size
                position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit: Price touches or crosses below Donchian low OR trend reversal
                if close[i] < donchian_low[i] or close[i] < ema50_1d_4h[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = position_size
            elif position == -1:
                # Exit: Price touches or crosses above Donchian high OR trend reversal
                if close[i] > donchian_high[i] or close[i] > ema50_1d_4h[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -position_size
    
    return signals