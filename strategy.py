#!/usr/bin/env python3
"""
Hypothesis: 6h Donchian(20) breakout with weekly pivot direction filter and volume confirmation.
Uses Donchian channel breakouts on 6h timeframe, filtered by weekly Camarilla pivot
direction (R3/S3 levels) to align with higher timeframe structure. Volume spike
confirms breakout momentum. Designed for 6h timeframe to capture medium-term moves
with controlled trade frequency (target: 12-37 trades/year per symbol).
Uses discrete position sizing (0.25) to manage drawdown and fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 6h Donchian channel (20-period)
    high_ma_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_ma_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d EMA50 for trend filter (secondary confirmation)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 1w Camarilla pivot levels (R3, S3) for weekly direction filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Typical price for Camarilla calculation
    typical_price_1w = (high_1w + low_1w + close_1w) / 3.0
    range_1w = high_1w - low_1w
    
    # Camarilla levels: R3 = close + (high-low)*1.1/4, S3 = close - (high-low)*1.1/4
    camarilla_r3_1w = close_1w + (range_1w * 1.1 / 4)
    camarilla_s3_1w = close_1w - (range_1w * 1.1 / 4)
    
    # Align weekly Camarilla levels to 6h timeframe (previous weekly bar values)
    camarilla_r3_1w_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r3_1w)
    camarilla_s3_1w_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s3_1w)
    
    # Calculate volume spike: current volume > 2.0x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 50)  # need Donchian20 and EMA50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(high_ma_20[i]) or np.isnan(low_ma_20[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(camarilla_r3_1w_aligned[i]) or 
            np.isnan(camarilla_s3_1w_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Donchian breakout conditions
        donchian_breakout_up = close[i] > high_ma_20[i-1]  # break above previous period high
        donchian_breakout_down = close[i] < low_ma_20[i-1]  # break below previous period low
        
        # Trend filters: price above/below 1d EMA50
        trend_up = close[i] > ema_50_1d_aligned[i]
        trend_down = close[i] < ema_50_1d_aligned[i]
        
        # Weekly direction filter: price relative to weekly Camarilla R3/S3
        weekly_bias_up = close[i] > camarilla_r3_1w_aligned[i]  # above weekly R3 = bullish bias
        weekly_bias_down = close[i] < camarilla_s3_1w_aligned[i]  # below weekly S3 = bearish bias
        
        if position == 0:
            # Long: Donchian breakout up AND uptrend AND weekly bullish bias AND volume spike
            if donchian_breakout_up and trend_up and weekly_bias_up and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Donchian breakout down AND downtrend AND weekly bearish bias AND volume spike
            elif donchian_breakout_down and trend_down and weekly_bias_down and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Donchian breakout in opposite direction
            exit_signal = False
            if position == 1:
                # Exit long on Donchian breakout down
                if donchian_breakout_down:
                    exit_signal = True
            elif position == -1:
                # Exit short on Donchian breakout up
                if donchian_breakout_up:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_Donchian20_Breakout_1wCamarillaR3S3_Direction_VolumeSpike"
timeframe = "6h"
leverage = 1.0