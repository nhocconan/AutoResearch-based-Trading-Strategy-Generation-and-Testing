#!/usr/bin/env python3
# 12h_Donchian20_Volume_Trend_v1
# Hypothesis: Donchian(20) breakout on 12h with volume confirmation and 1d EMA trend filter
# captures sustained moves in both bull and bear markets. Low-frequency entries
# (target 20-50 trades/year) minimize fee drag. Volume spike confirms breakout
# strength, while 1d EMA50 ensures alignment with higher-timeframe trend.

name = "12h_Donchian20_Volume_Trend_v1"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data for EMA trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 12h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- Donchian Channel (20-period) on 12h ---
    # Upper band: 20-period high
    # Lower band: 20-period low
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # --- 1d EMA50 Trend Filter ---
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # --- Volume Spike (12h) ---
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_spike = volume > (2.0 * vol_ma.values)  # 2x average volume
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if EMA is not ready
        if np.isnan(ema50_1d_aligned[i]):
            if position != 0:
                # Hold position until trend filter is ready
                signals[i] = 0.25 if position == 1 else -0.25
            continue
        
        # Entry conditions
        # Long: price breaks above Donchian high + volume spike + price above 1d EMA50
        long_entry = (close[i] > donchian_high[i-1]) and vol_spike[i] and (close[i] > ema50_1d_aligned[i])
        # Short: price breaks below Donchian low + volume spike + price below 1d EMA50
        short_entry = (close[i] < donchian_low[i-1]) and vol_spike[i] and (close[i] < ema50_1d_aligned[i])
        
        if position == 0:
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit long if price breaks below Donchian low or trend fails
                if (close[i] < donchian_low[i-1]) or (close[i] < ema50_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short if price breaks above Donchian high or trend fails
                if (close[i] > donchian_high[i-1]) or (close[i] > ema50_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals