#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_1dATR_Trend_VolumeConfirm
Hypothesis: Donchian(20) channel breakouts capture institutional moves when aligned with 1d ATR-based trend and volume spike.
Only trade in direction of 1d ATR trend (price above/below ATR-adjusted mean). Volume confirmation filters false breakouts.
Designed for 75-200 trades over 4 years on 4h timeframe. Works in bull/bear via 1d ATR trend filter.
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
    
    # 1d data for ATR trend filter and Donchian context (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    
    # 1d ATR(14) for trend definition
    tr1 = pd.Series(df_1d['high']).diff().abs()
    tr2 = (pd.Series(df_1d['high']) - pd.Series(df_1d['close'].shift())).abs()
    tr3 = (pd.Series(df_1d['low']) - pd.Series(df_1d['close'].shift())).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14 = tr.rolling(window=14, min_periods=14).mean().values
    
    # 1d ATR-adjusted mean: close ± 0.5*ATR for trend bias
    atr_mean = pd.Series(df_1d['close']).rolling(window=50, min_periods=50).mean().values
    atr_trend_up = atr_mean + (0.5 * atr_14)
    atr_trend_down = atr_mean - (0.5 * atr_14)
    atr_trend_up_aligned = align_htf_to_ltf(prices, df_1d, atr_trend_up)
    atr_trend_down_aligned = align_htf_to_ltf(prices, df_1d, atr_trend_down)
    
    # 4h Donchian(20) channels
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.8 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Donchian (20) + volume MA (20) + ATR (50)
    start_idx = max(20, 20, 50) + 5
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(atr_trend_up_aligned[i]) or np.isnan(atr_trend_down_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        
        if position == 0:
            # Look for entry signals - require: Donchian breakout + volume spike + ATR trend alignment
            long_entry = (curr_close > donchian_high[i]) and volume_spike[i] and (curr_close > atr_trend_up_aligned[i])
            short_entry = (curr_close < donchian_low[i]) and volume_spike[i] and (curr_close < atr_trend_down_aligned[i])
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position: exit when price closes below Donchian low or trend turns bearish
            if curr_close < donchian_low[i] or curr_close < atr_trend_down_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit when price closes above Donchian high or trend turns bullish
            if curr_close > donchian_high[i] or curr_close > atr_trend_up_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Breakout_1dATR_Trend_VolumeConfirm"
timeframe = "4h"
leverage = 1.0