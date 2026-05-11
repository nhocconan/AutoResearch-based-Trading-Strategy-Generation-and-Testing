#!/usr/bin/env python3
# 4H_Donchian20_VolumeTrend_v2
# Hypothesis: Donchian(20) breakout on 4h with volume confirmation and 1d EMA50 trend filter provides robust trend-following signals in both bull and bear markets.
# Entry requires price breaking above/below Donchian channel, volume > 1.5x 20-period average, and alignment with 1d EMA50 trend.
# Exit on Donchian opposite breakout or volume drop. Designed for low frequency (~25-50 trades/year) to minimize fee drag.

name = "4H_Donchian20_VolumeTrend_v2"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data for EMA50 (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 4h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- Donchian Channel (20-period) on 4h ---
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max()
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min()
    donchian_up = highest_high.values
    donchian_low = lowest_low.values
    
    # --- 1d EMA50 ---
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # --- Volume Spike (4h) ---
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_spike = volume > (1.5 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if EMA50 is NaN
        if np.isnan(ema50_1d_aligned[i]):
            if position != 0:
                # Simple exit: reverse Donchian breakout
                if position == 1 and close[i] < donchian_low[i]:
                    signals[i] = 0.0
                    position = 0
                elif position == -1 and close[i] > donchian_up[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25 if position == 1 else -0.25
            continue
        
        # Entry conditions
        long_breakout = close[i] > donchian_up[i]
        short_breakout = close[i] < donchian_low[i]
        long_entry = long_breakout and vol_spike[i] and (close[i] > ema50_1d_aligned[i])
        short_entry = short_breakout and vol_spike[i] and (close[i] < ema50_1d_aligned[i])
        
        if position == 0:
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
        else:
            # Exit on opposite Donchian breakout or volume drop
            if position == 1:
                if close[i] < donchian_low[i] or not vol_spike[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                if close[i] > donchian_up[i] or not vol_spike[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals