#!/usr/bin/env python3
# 4h_RSI_2_Trend_Volume
# Hypothesis: RSI(2) on 4h with extreme thresholds (<10 for long, >90 for short) + 1d EMA trend filter + volume confirmation.
# Works in both bull and bear markets by capturing mean-reversion extremes in strong trends.
# Low trade frequency due to strict RSI thresholds and trend alignment requirement.

name = "4h_RSI_2_Trend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA34 for trend filter
    ema34_1d = np.full_like(close_1d, np.nan)
    if len(close_1d) >= 34:
        ema34_1d[33] = np.mean(close_1d[0:34])
        for i in range(34, len(close_1d)):
            ema34_1d[i] = (close_1d[i] * 2 + ema34_1d[i-1] * 32) / 34
    
    # Calculate RSI(2) on 4h close
    rsi2 = np.full_like(close, np.nan)
    if len(close) >= 2:
        change = np.diff(close, prepend=close[0])
        gain = np.where(change > 0, change, 0.0)
        loss = np.where(change < 0, -change, 0.0)
        
        avg_gain = np.full_like(close, np.nan)
        avg_loss = np.full_like(close, np.nan)
        
        avg_gain[1] = gain[1]
        avg_loss[1] = loss[1]
        
        for i in range(2, len(close)):
            avg_gain[i] = (avg_gain[i-1] * 1 + gain[i]) / 2
            avg_loss[i] = (avg_loss[i-1] * 1 + loss[i]) / 2
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
        rsi2 = 100 - (100 / (1 + rs))
        rsi2[avg_loss == 0] = 100
        rsi2[avg_gain == 0] = 0
    
    # Volume filter: current volume vs 20-period average
    vol_ma = np.full_like(volume, np.nan)
    if len(volume) >= 20:
        vol_ma[19] = np.mean(volume[0:20])
        for i in range(20, len(volume)):
            vol_ma[i] = (vol_ma[i-1] * 19 + volume[i]) / 20
    
    volume_ratio = np.full_like(volume, np.nan)
    valid_vol = (~np.isnan(vol_ma)) & (vol_ma != 0)
    volume_ratio[valid_vol] = volume[valid_vol] / vol_ma[valid_vol]
    
    # Align 1d EMA to 4h timeframe
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 2, 20)  # Need 1d EMA, RSI(2), and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(rsi2[i]) or np.isnan(volume_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine 1d trend
        trend_up = close[i] > ema34_1d_aligned[i]
        
        if position == 0:
            # Enter long: 1d trend up + RSI(2) < 10 + volume confirmation
            if trend_up and rsi2[i] < 10 and volume_ratio[i] > 1.5:
                signals[i] = 0.25
                position = 1
            # Enter short: 1d trend down + RSI(2) > 90 + volume confirmation
            elif not trend_up and rsi2[i] > 90 and volume_ratio[i] > 1.5:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: RSI(2) > 50 (mean reversion complete) or trend turns down
            if rsi2[i] > 50 or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: RSI(2) < 50 (mean reversion complete) or trend turns up
            if rsi2[i] < 50 or trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals