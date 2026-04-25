#!/usr/bin/env python3
"""
6h_Donchian20_Breakout_12hTrend_1dVolumeConfirm_v1
Hypothesis: Trade 6h Donchian(20) breakouts aligned with 12h EMA50 trend and confirmed by 1d volume spike (volume > 1.5x 20-period average).
Donchian breakouts capture momentum; 12h EMA50 filters for higher-timeframe trend alignment; 1d volume spike adds conviction.
Only long when price > upper band + 12h uptrend + volume spike; only short when price < lower band + 12h downtrend + volume spike.
Exit on opposite band touch or trend reversal.
Position size: 0.25 to balance profit and fee drag.
Target: 12-30 trades/year (50-120 over 4 years) to stay well under 300-trade 6h hard max.
Works in bull (breakouts with trend) and bear (strong breakdowns with trend) markets.
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
    
    # Get 12h data for HTF trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for HTF trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Get 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d volume spike: current 1d volume > 1.5x 20-period average
    vol_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike = vol_1d > (1.5 * vol_ma_20)
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike.astype(float))  # bool -> float for alignment
    
    # Calculate 6h Donchian(20) channels
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for Donchian (20), 12h EMA50 (50), 1d vol MA (20)
    start_idx = max(lookback, 50, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(vol_spike_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine 12h HTF trend (bullish = price above EMA50)
        htf_12h_bullish = close[i] > ema_50_12h_aligned[i]
        htf_12h_bearish = close[i] < ema_50_12h_aligned[i]
        
        # Volume confirmation: 1d volume spike
        vol_confirmed = vol_spike_aligned[i] > 0.5  # treated as boolean
        
        if position == 0:
            # Long setup: price breaks above upper Donchian band + 12h uptrend + volume spike
            long_setup = (close[i] > highest_high[i]) and htf_12h_bullish and vol_confirmed
            
            # Short setup: price breaks below lower Donchian band + 12h downtrend + volume spike
            short_setup = (close[i] < lowest_low[i]) and htf_12h_bearish and vol_confirmed
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: price touches lower Donchian band (stop) OR 12h trend turns bearish
            if (close[i] <= lowest_low[i]) or (not htf_12h_bullish):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price touches upper Donchian band (stop) OR 12h trend turns bullish
            if (close[i] >= highest_high[i]) or (htf_12h_bullish):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Donchian20_Breakout_12hTrend_1dVolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0