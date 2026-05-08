#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 12h EMA50 trend filter + volume confirmation
# Uses 12h EMA50 for trend direction, Donchian channel breakout for entry timing,
# and volume spike (>1.5x average) for confirmation. Designed to capture trends
# in both bull and bear markets by following the 12h trend while avoiding false
# breakouts in low-volume conditions. Target: 25-40 trades/year.

name = "4h_Donchian20_12hEMA50_VolumeConfirm"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema50_12h = np.full(len(close_12h), np.nan)
    if len(close_12h) >= 50:
        ema50_12h[49] = np.mean(close_12h[:50])
        for i in range(50, len(close_12h)):
            ema50_12h[i] = (close_12h[i] * 2 + ema50_12h[i-1] * 48) / 50
    
    # Calculate 12h ATR for volatility filter
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    tr_12h = np.maximum(high_12h[1:] - low_12h[1:], 
                        np.maximum(np.abs(high_12h[1:] - close_12h[:-1]),
                                   np.abs(low_12h[1:] - close_12h[:-1])))
    tr_12h = np.concatenate([[np.nan], tr_12h])
    atr14_12h = np.full(len(tr_12h), np.nan)
    if len(tr_12h) >= 14:
        atr14_12h[13] = np.nanmean(tr_12h[:14])
        for i in range(14, len(tr_12h)):
            atr14_12h[i] = (atr14_12h[i-1] * 13 + tr_12h[i]) / 14
    
    # Calculate 20-period Donchian channels (highest high, lowest low)
    highest_high_20 = np.full(len(close), np.nan)
    lowest_low_20 = np.full(len(close), np.nan)
    if len(close) >= 20:
        for i in range(20, len(close)):
            highest_high_20[i] = np.max(high[i-20:i])
            lowest_low_20[i] = np.min(low[i-20:i])
    
    # Calculate volume average for volume confirmation
    vol_avg_20 = np.full(len(volume), np.nan)
    if len(volume) >= 20:
        for i in range(20, len(volume)):
            vol_avg_20[i] = np.mean(volume[i-20:i])
    
    # Align 12h indicators to 4h timeframe
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    atr14_12h_aligned = align_htf_to_ltf(prices, df_12h, atr14_12h)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(ema50_12h_aligned[i]) or np.isnan(atr14_12h_aligned[i]) or
            np.isnan(highest_high_20[i]) or np.isnan(lowest_low_20[i]) or
            np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume[i] > 1.5 * vol_avg_20[i]
        
        if position == 0:
            # Look for entry: Donchian breakout in direction of 12h EMA trend
            # Long when price breaks above upper Donchian band in uptrend
            long_condition = (
                close[i] > highest_high_20[i] and   # breakout above Donchian high
                close[i] > ema50_12h_aligned[i] and # price above 12h EMA50 (uptrend)
                vol_confirm                         # volume confirmation
            )
            
            # Short when price breaks below lower Donchian band in downtrend
            short_condition = (
                close[i] < lowest_low_20[i] and     # breakdown below Donchian low
                close[i] < ema50_12h_aligned[i] and # price below 12h EMA50 (downtrend)
                vol_confirm                         # volume confirmation
            )
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns below 12h EMA50 or breaks below lower Donchian
            if close[i] < ema50_12h_aligned[i] or close[i] < lowest_low_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns above 12h EMA50 or breaks above upper Donchian
            if close[i] > ema50_12h_aligned[i] or close[i] > highest_high_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals