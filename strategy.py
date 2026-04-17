#!/usr/bin/env python3
"""
Hypothesis: 1h Donchian channel breakout with 4h EMA100 trend filter and volume confirmation.
Uses Donchian(20) for breakout detection, 4h EMA100 for trend filter, and volume spike (>1.5x 20-period average) for confirmation.
Designed to capture strong trending moves while avoiding false breakouts in choppy markets.
Target: 15-37 trades/year with low turnover to minimize fee drag. Works in both bull and bear markets by using trend filter.
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
    
    # === Donchian(20) on 1h ===
    lookback = 20
    donchian_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    donchian_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # === 4h EMA100 (trend filter) ===
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    ema100_4h = pd.Series(close_4h).ewm(span=100, adjust=False, min_periods=100).mean().values
    ema100_4h_aligned = align_htf_to_ltf(prices, df_4h, ema100_4h)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_20_4h)
    
    signals = np.zeros(n)
    
    # Warmup: need enough data for Donchian and EMA
    warmup = max(lookback, 100)
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(ema100_4h_aligned[i]) or np.isnan(vol_ma_20_4h_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Get aligned volume for current bar
        vol_current_aligned = align_htf_to_ltf(prices, df_4h, volume_4h)
        
        # Volume spike condition
        vol_spike = vol_current_aligned[i] > vol_ma_20_4h_aligned[i] * 1.5
        
        # Trend filter: price relative to 4h EMA100
        price_above_ema100 = close[i] > ema100_4h_aligned[i]
        price_below_ema100 = close[i] < ema100_4h_aligned[i]
        
        # Breakout conditions
        breakout_up = close[i] > donchian_high[i]
        breakout_down = close[i] < donchian_low[i]
        
        # Entry logic: only enter when flat
        if position == 0:
            if vol_spike:
                # Long: upward breakout AND price above 4h EMA100
                if breakout_up and price_above_ema100:
                    signals[i] = 0.20
                    position = 1
                    continue
                # Short: downward breakout AND price below 4h EMA100
                elif breakout_down and price_below_ema100:
                    signals[i] = -0.20
                    position = -1
                    continue
        
        # Exit logic: exit when opposite breakout occurs or trend changes
        elif position == 1:
            # Exit long if downward breakout or price crosses below EMA100
            if breakout_down or close[i] < ema100_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short if upward breakout or price crosses above EMA100
            if breakout_up or close[i] > ema100_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Donchian20_4hEMA100_VolumeSpike_1.5x"
timeframe = "1h"
leverage = 1.0