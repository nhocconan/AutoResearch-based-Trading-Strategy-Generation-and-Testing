#!/usr/bin/env python3
# 4h_4h_Donchian20_1dTrend_Volume_Confirm_V1
# Hypothesis: 4h Donchian(20) breakout with 1d trend filter (EMA200) and volume confirmation.
# Works in bull (breakout continuation) and bear (breakdown continuation) by filtering with higher timeframe trend.
# Target: ~25-40 trades/year per symbol (100-160 total over 4 years) to avoid fee drag.
# Uses 1d EMA200 for trend, volume > 1.5x 20-period average for confirmation.

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
    
    # Get daily data for trend filter and volume average
    df_1d = get_htf_data(prices, '1d')
    
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Daily EMA200 for trend filter
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Daily volume average (20-period)
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align daily data to 4h timeframe
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # need enough for EMA200
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if np.isnan(ema_200_aligned[i]) or np.isnan(vol_ma_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Trend condition from 1d EMA200
        uptrend = close[i] > ema_200_aligned[i]
        downtrend = close[i] < ema_200_aligned[i]
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma_aligned[i]
        
        # Donchian(20) breakout levels (using last 20 periods, including current)
        if i >= 20:
            highest_high = np.max(high[i-19:i+1])  # last 20 highs including current
            lowest_low = np.min(low[i-19:i+1])    # last 20 lows including current
        else:
            highest_high = np.max(high[:i+1])
            lowest_low = np.min(low[:i+1])
        
        breakout_up = close[i] > highest_high
        breakdown_down = close[i] < lowest_low
        
        if position == 0:
            # Long: uptrend + volume + breakout above Donchian high
            if uptrend and vol_confirm and breakout_up:
                signals[i] = 0.25
                position = 1
            # Short: downtrend + volume + breakdown below Donchian low
            elif downtrend and vol_confirm and breakdown_down:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: trend change, volume confirmation, or breakdown below Donchian low
            if not uptrend or (vol_confirm and breakdown_down):
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: trend change, volume confirmation, or breakout above Donchian high
            if not downtrend or (vol_confirm and breakout_up):
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_1dTrend_Volume_Confirm_V1"
timeframe = "4h"
leverage = 1.0