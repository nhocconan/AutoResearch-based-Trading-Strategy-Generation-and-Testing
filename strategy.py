#!/usr/bin/env python3
name = "12h_Donchian_Breakout_1dTrend_VolumeSqueeze"
timeframe = "12h"
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
    
    # === 1d Data for trend and Donchian channels ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # === 1d EMA34 for trend ===
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # === 1d Donchian(20) channels ===
    lookback = 20
    high_roll = pd.Series(high_1d).rolling(window=lookback, min_periods=lookback).max().values
    low_roll = pd.Series(low_1d).rolling(window=lookback, min_periods=lookback).min().values
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, high_roll)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, low_roll)
    
    # === Volume squeeze detection: Bollinger Band width percentile ===
    bb_window = 20
    bb_std_dev = 2.0
    sma = pd.Series(close_1d).rolling(window=bb_window, min_periods=bb_window).mean().values
    std_dev = pd.Series(close_1d).rolling(window=bb_window, min_periods=bb_window).std().values
    upper_bb = sma + (bb_std_dev * std_dev)
    lower_bb = sma - (bb_std_dev * std_dev)
    bb_width = upper_bb - lower_bb
    bb_width_percentile = pd.Series(bb_width).rolling(window=50, min_periods=10).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else 0.5, raw=False
    ).values
    bb_width_percentile_aligned = align_htf_to_ltf(prices, df_1d, bb_width_percentile)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20, 50)  # EMA34, Donchian20, BBwidth50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(donchian_high_aligned[i]) or
            np.isnan(donchian_low_aligned[i]) or
            np.isnan(bb_width_percentile_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Donchian breakout above + uptrend + low volatility (squeeze)
            if (close[i] > donchian_high_aligned[i] and
                close[i] > ema34_1d_aligned[i] and
                bb_width_percentile_aligned[i] < 0.3):  # Low volatility squeeze
                signals[i] = 0.25
                position = 1
            # Short: Donchian breakdown below + downtrend + low volatility (squeeze)
            elif (close[i] < donchian_low_aligned[i] and
                  close[i] < ema34_1d_aligned[i] and
                  bb_width_percentile_aligned[i] < 0.3):  # Low volatility squeeze
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Donchian breakdown or trend change
            if close[i] < donchian_low_aligned[i] or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Donchian breakout or trend change
            if close[i] > donchian_high_aligned[i] or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals