#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h with 4h trend filter and volume confirmation
# Uses 4h EMA21 for trend direction and volume > 1.5x 20-period average for entry.
# Designed to capture trending moves while avoiding choppy markets.
# Target: 15-35 trades/year per symbol, works in both bull and bear markets.

name = "1h_EMA21_VolumeBreakout_4hTrend"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 21:
        return np.zeros(n)
    
    # Calculate 4h EMA21 for trend filter
    close_4h = df_4h['close'].values
    ema21_4h = np.full(len(close_4h), np.nan)
    if len(close_4h) >= 21:
        ema21_4h[20] = np.mean(close_4h[:21])
        for i in range(21, len(close_4h)):
            ema21_4h[i] = (close_4h[i] * 2 + ema21_4h[i-1] * 19) / 21
    
    # Calculate 20-period average volume for 4h
    vol_4h = df_4h['volume'].values
    vol_avg_20_4h = np.full(len(vol_4h), np.nan)
    if len(vol_4h) >= 20:
        for i in range(20, len(vol_4h)):
            vol_avg_20_4h[i] = np.mean(vol_4h[i-20:i])
    
    # Align 4h indicators to 1h timeframe
    ema21_4h_aligned = align_htf_to_ltf(prices, df_4h, ema21_4h)
    vol_avg_20_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_avg_20_4h)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(21, 20)  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if any required data is NaN
        if np.isnan(ema21_4h_aligned[i]) or np.isnan(vol_avg_20_4h_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume breakout: current 1h volume > 1.5x 20-period average of 4h volume
        vol_breakout = volume[i] > 1.5 * vol_avg_20_4h_aligned[i]
        
        if position == 0:
            # Look for entry: follow 4h EMA trend with volume confirmation
            long_condition = (
                close[i] > ema21_4h_aligned[i] and   # price above 4h EMA21 (bullish bias)
                vol_breakout                         # volume breakout for entry
            )
            
            short_condition = (
                close[i] < ema21_4h_aligned[i] and   # price below 4h EMA21 (bearish bias)
                vol_breakout                         # volume breakout for entry
            )
            
            if long_condition:
                signals[i] = 0.20
                position = 1
            elif short_condition:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price returns below 4h EMA21
            if close[i] < ema21_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price returns above 4h EMA21
            if close[i] > ema21_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals