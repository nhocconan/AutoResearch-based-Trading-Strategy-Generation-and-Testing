#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA34 trend filter and volume confirmation.
Long when price breaks above 4h Camarilla R3 level AND 4h close > 4h EMA34 (uptrend) AND volume > 1.5x 20-period MA.
Short when price breaks below 4h Camarilla S3 level AND 4h close < 4h EMA34 (downtrend) AND volume > 1.5x 20-period MA.
Exit when price retraces to the 4h Camarilla pivot point or 4h trend reverses.
Uses higher Camarilla levels (R3/S3) for fewer, more significant breakouts.
Target: 15-35 trades/year to avoid fee drag on 1h timeframe.
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
    
    # Calculate 4h Camarilla levels (R3, S3, pivot) from previous bar
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_hl = prev_high - prev_low
    camarilla_r3 = pivot + range_hl * 1.1 / 4.0   # R3 = pivot + (high-low)*1.1/4
    camarilla_s3 = pivot - range_hl * 1.1 / 4.0   # S3 = pivot - (high-low)*1.1/4
    
    # Calculate 4h EMA34 for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # Calculate 4h close aligned for trend comparison
    close_4h_aligned = align_htf_to_ltf(prices, df_4h, close_4h)
    
    # Calculate 1h volume MA (20-period) for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(pivot[i]) or np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or
            np.isnan(ema_34_4h_aligned[i]) or np.isnan(close_4h_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: 4h close > EMA34 = uptrend, close < EMA34 = downtrend
        trend_up = close_4h_aligned[i] > ema_34_4h_aligned[i]
        trend_down = close_4h_aligned[i] < ema_34_4h_aligned[i]
        
        # Volume filter: 1h volume > 1.5x 20-period MA
        vol_filter = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Long: price breaks above Camarilla R3 AND uptrend AND volume filter
            if close[i] > camarilla_r3[i] and trend_up and vol_filter:
                signals[i] = 0.20
                position = 1
            # Short: price breaks below Camarilla S3 AND downtrend AND volume filter
            elif close[i] < camarilla_s3[i] and trend_down and vol_filter:
                signals[i] = -0.20
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: price retraces to Camarilla pivot OR 4h trend turns down
                if close[i] <= pivot[i] or not trend_up:
                    exit_signal = True
            elif position == -1:
                # Short exit: price retraces to Camarilla pivot OR 4h trend turns up
                if close[i] >= pivot[i] or not trend_down:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1H_Camarilla_R3S3_Breakout_4hEMA34_Trend_VolumeFilter"
timeframe = "1h"
leverage = 1.0