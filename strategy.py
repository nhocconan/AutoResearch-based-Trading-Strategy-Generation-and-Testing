#!/usr/bin/env python3
# 4h_Three_Bar_Reversal_1dTrend_Volume
# Hypothesis: Three-bar reversal pattern (bullish/bearish) on 4h timeframe confirmed by 1d trend and volume.
# Bullish 3-bar reversal: low[0] > low[1] > low[2] and close[0] > close[1] > close[2]
# Bearish 3-bar reversal: high[0] < high[1] < high[2] and close[0] < close[1] < close[2]
# Trades only in direction of 1d trend (EMA50) to avoid counter-trend whipsaws.
# Volume > 1.5x 20-bar average confirms momentum. Designed for ~25-40 trades/year.

name = "4h_Three_Bar_Reversal_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # Calculate 1d EMA(50)
    ema_50_1d = np.full_like(close_1d, np.nan)
    if len(close_1d) >= 50:
        ema_50_1d[49] = np.mean(close_1d[0:50])
        for i in range(50, len(close_1d)):
            ema_50_1d[i] = (close_1d[i] * 2 + ema_50_1d[i-1] * 48) / 50
    
    # Align 1d EMA to 4h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume filter: volume / 20-period average
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
    
    start_idx = max(20, 2)  # Need at least 2 bars for pattern
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_ratio[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if i >= 2:  # Need 3 bars for pattern
            # Bullish 3-bar reversal: higher lows and higher closes
            bullish_reversal = (low[i] > low[i-1] > low[i-2]) and (close[i] > close[i-1] > close[i-2])
            # Bearish 3-bar reversal: lower highs and lower closes
            bearish_reversal = (high[i] < high[i-1] < high[i-2]) and (close[i] < close[i-1] < close[i-2])
        else:
            bullish_reversal = False
            bearish_reversal = False
        
        if position == 0:
            # Enter long: bullish reversal + volume confirmation + uptrend (price > 1d EMA50)
            if bullish_reversal and volume_ratio[i] > 1.5 and close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: bearish reversal + volume confirmation + downtrend (price < 1d EMA50)
            elif bearish_reversal and volume_ratio[i] > 1.5 and close[i] < ema_50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: bearish reversal or trend turns down
            if bearish_reversal or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: bullish reversal or trend turns up
            if bullish_reversal or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals