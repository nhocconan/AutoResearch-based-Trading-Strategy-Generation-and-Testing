#!/usr/bin/env python3
"""
6h_Weekly_Pivot_Direction_Donchian_Breakout_4hTrend_VolumeConfirm
Hypothesis: 6h Donchian(20) breakout in direction of weekly pivot trend (price vs weekly pivot) with 4h EMA50 trend filter and volume confirmation.
Long when price breaks above Donchian(20) high AND price > weekly pivot AND price > 4h EMA50 AND volume spike (>1.5x avg).
Short when price breaks below Donchian(20) low AND price < weekly pivot AND price < 4h EMA50 AND volume spike.
Exit on opposite Donchian level break or loss of 4h EMA50 alignment.
Designed for 12-30 trades/year on 6h to minimize fee drag while capturing strong intraday moves aligned with weekly structure.
Works in bull markets (breakouts with weekly uptrend) and bear markets (breakdowns with weekly downtrend).
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
    
    # Calculate weekly pivot from prior week (1w)
    df_1w = get_htf_data(prices, '1w')
    # Prior week OHLC (shifted by 1 to avoid look-ahead)
    prev_week_close = pd.Series(df_1w['close'].values).shift(1)
    prev_week_high = pd.Series(df_1w['high'].values).shift(1)
    prev_week_low = pd.Series(df_1w['low'].values).shift(1)
    
    # Weekly pivot point (standard calculation)
    weekly_pivot = (prev_week_high + prev_week_low + prev_week_close) / 3.0
    
    # Align weekly pivot to 6h
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot.values)
    
    # Donchian(20) channels from 6h data
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 4h EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    ema_50 = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_4h, ema_50)
    
    # Volume spike: current volume > 1.5 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    size = 0.25  # 25% position size
    
    # Warmup: need enough for 1w pivot (2w), Donchian(20), 4h EMA50 (~20 6h bars), volume avg
    start_idx = max(40, 20, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(weekly_pivot_aligned[i]) or np.isnan(donchian_high[i]) or
            np.isnan(donchian_low[i]) or np.isnan(ema_50_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        pivot_val = weekly_pivot_aligned[i]
        upper_donchian = donchian_high[i]
        lower_donchian = donchian_low[i]
        ema_val = ema_50_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Flat - look for entry: Donchian breakout with weekly pivot alignment, 4h EMA50 trend, and volume spike
            # Long: Close > Donchian upper AND price > weekly pivot AND price > 4h EMA50 AND volume spike
            # Short: Close < Donchian lower AND price < weekly pivot AND price < 4h EMA50 AND volume spike
            long_condition = (close_val > upper_donchian and 
                            close_val > pivot_val and 
                            close_val > ema_val and 
                            vol_spike)
            short_condition = (close_val < lower_donchian and 
                             close_val < pivot_val and 
                             close_val < ema_val and 
                             vol_spike)
            
            if long_condition:
                signals[i] = size
                position = 1
                entry_price = close_val
            elif short_condition:
                signals[i] = -size
                position = -1
                entry_price = close_val
        elif position == 1:
            # Long - exit when price breaks below Donchian lower OR loses 4h EMA50 alignment
            if close_val < lower_donchian or close_val < ema_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit when price breaks above Donchian upper OR loses 4h EMA50 alignment
            if close_val > upper_donchian or close_val > ema_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Weekly_Pivot_Direction_Donchian_Breakout_4hTrend_VolumeConfirm"
timeframe = "6h"
leverage = 1.0