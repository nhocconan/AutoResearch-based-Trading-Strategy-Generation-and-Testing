#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_1dTrendFilter_VolumeConfirm_v1
Hypothesis: Donchian(20) breakout on 4h with 1d EMA50 trend filter and volume confirmation captures strong directional moves while avoiding false breakouts in chop. Works in bull/bear by only taking breakouts aligned with daily trend. Targets 20-50 trades/year on 4h with discrete sizing (0.25).
"""

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
    
    # Load 1d data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 4h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Donchian(20) on 4h: highest high and lowest low of past 20 bars
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Warmup: max of Donchian (20), EMA50 (50), volume MA (20)
    start_idx = max(20, 50, 20)
    
    for i in range(start_idx, n):
        close_val = close[i]
        donch_high = donchian_high[i]
        donch_low = donchian_low[i]
        ema_trend = ema50_1d_aligned[i]
        vol_conf = volume_confirm[i]
        
        # Skip if any data not ready
        if (np.isnan(donch_high) or np.isnan(donch_low) or np.isnan(ema_trend)):
            # Hold current position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        # Trend filter: only long if price > daily EMA50, short if price < daily EMA50
        is_uptrend = close_val > ema_trend
        is_downtrend = close_val < ema_trend
        
        # Entry conditions
        long_condition = (close_val > donch_high) and is_uptrend and vol_conf
        short_condition = (close_val < donch_low) and is_downtrend and vol_conf
        
        # Exit conditions: opposite Donchian breakout or loss of trend alignment
        long_exit = (position == 1 and (close_val < donch_low or not is_uptrend))
        short_exit = (position == -1 and (close_val > donch_high or not is_downtrend))
        
        if long_condition and position != 1:
            signals[i] = base_size
            position = 1
        elif short_condition and position != -1:
            signals[i] = -base_size
            position = -1
        elif long_exit:
            signals[i] = 0.0
            position = 0
        elif short_exit:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
    
    return signals

name = "4h_Donchian20_Breakout_1dTrendFilter_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0