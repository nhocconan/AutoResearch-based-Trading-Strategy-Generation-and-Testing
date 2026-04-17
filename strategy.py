#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian channel breakout with 12h trend filter and volume confirmation.
Uses Donchian(20) for breakout detection, 12h EMA50 for trend filter, and volume spike (>1.5x 20-period average) for confirmation.
Designed to capture strong trending moves while avoiding false breakouts in choppy markets.
Target: 20-50 trades/year with low turnover to minimize fee drag. Works in both bull and bear markets by using trend filter.
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
    
    # === Donchian(20) on 4h ===
    lookback = 20
    donchian_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    donchian_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # === 12h EMA50 (trend filter) ===
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_20_12h)
    
    signals = np.zeros(n)
    
    # Warmup: need enough data for Donchian and EMA
    warmup = max(lookback, 50)
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(ema50_12h_aligned[i]) or np.isnan(vol_ma_20_12h_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Get aligned volume for current bar
        vol_current_aligned = align_htf_to_ltf(prices, df_12h, volume_12h)
        
        # Volume spike condition
        vol_spike = vol_current_aligned[i] > vol_ma_20_12h_aligned[i] * 1.5
        
        # Trend filter: price relative to 12h EMA50
        price_above_ema50 = close[i] > ema50_12h_aligned[i]
        price_below_ema50 = close[i] < ema50_12h_aligned[i]
        
        # Breakout conditions
        breakout_up = close[i] > donchian_high[i]
        breakout_down = close[i] < donchian_low[i]
        
        # Entry logic: only enter when flat
        if position == 0:
            if vol_spike:
                # Long: upward breakout AND price above 12h EMA50
                if breakout_up and price_above_ema50:
                    signals[i] = 0.25
                    position = 1
                    continue
                # Short: downward breakout AND price below 12h EMA50
                elif breakout_down and price_below_ema50:
                    signals[i] = -0.25
                    position = -1
                    continue
        
        # Exit logic: exit when opposite breakout occurs or trend changes
        elif position == 1:
            # Exit long if downward breakout or price crosses below EMA50
            if breakout_down or close[i] < ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short if upward breakout or price crosses above EMA50
            if breakout_up or close[i] > ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_12hEMA50_VolumeSpike_1.5x"
timeframe = "4h"
leverage = 1.0