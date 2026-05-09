#!/usr/bin/env python3
# 1d_HighLow_Breakout_Trend_Volume
# Hypothesis: Breakouts above the daily high or below the daily low with trend filter (1-week EMA) and volume confirmation.
# Designed to capture strong directional moves in both bull and bear markets by combining price breakouts with trend alignment.
# Uses weekly EMA for trend direction and daily volume spike for confirmation.
# Target: 15-25 trades per year per symbol with controlled risk.

name = "1d_HighLow_Breakout_Trend_Volume"
timeframe = "1d"
leverage = 1.0

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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA(20) for trend
    ema_20_1w = np.full_like(close_1w, np.nan)
    if len(close_1w) >= 20:
        ema_20_1w[19] = np.mean(close_1w[0:20])
        alpha = 2 / (20 + 1)
        for i in range(20, len(close_1w)):
            ema_20_1w[i] = alpha * close_1w[i] + (1 - alpha) * ema_20_1w[i-1]
    
    # Align weekly EMA to daily timeframe
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Calculate daily high and low for breakout levels
    # Use previous day's high/low for breakout detection
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_high[0] = np.nan  # First day has no previous
    prev_low[0] = np.nan
    
    # Volume filter: current volume / 20-day average volume
    vol_ma = np.full_like(volume, np.nan)
    if len(volume) >= 20:
        vol_ma[19] = np.mean(volume[0:20])
        for i in range(20, len(volume)):
            vol_ma[i] = (vol_ma[i-1] * 19 + volume[i]) / 20
    
    volume_ratio = np.full_like(volume, np.nan)
    valid_vol = (~np.isnan(vol_ma)) & (vol_ma != 0)
    volume_ratio[valid_vol] = volume[valid_vol] / vol_ma[valid_vol]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 1)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(ema_20_1w_aligned[i]) or np.isnan(prev_high[i]) or np.isnan(prev_low[i]) or \
           np.isnan(volume_ratio[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: price above/below weekly EMA
        uptrend = close[i] > ema_20_1w_aligned[i]
        downtrend = close[i] < ema_20_1w_aligned[i]
        
        if position == 0:
            # Enter long: Price breaks above previous day's high AND uptrend AND volume confirmation
            if close[i] > prev_high[i] and uptrend and volume_ratio[i] > 2.0:
                signals[i] = 0.25
                position = 1
            # Enter short: Price breaks below previous day's low AND downtrend AND volume confirmation
            elif close[i] < prev_low[i] and downtrend and volume_ratio[i] > 2.0:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price breaks below previous day's low OR trend reverses
            if close[i] < prev_low[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price breaks above previous day's high OR trend reverses
            if close[i] > prev_high[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals