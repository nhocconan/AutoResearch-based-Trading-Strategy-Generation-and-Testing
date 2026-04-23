#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla pivot R1/S1 breakout with 12h EMA34 trend filter and volume spike confirmation.
Long when price breaks above 4h Camarilla R1 level AND 12h close > 12h EMA34 (uptrend) AND volume > 1.8x 20-period MA.
Short when price breaks below 4h Camarilla S1 level AND 12h close < 12h EMA34 (downtrend) AND volume > 1.8x 20-period MA.
Exit when price retouches the opposite Camarilla level (S1 for long, R1 for short) or 12h trend reverses.
Camarilla levels provide precise intraday support/resistance; 12h EMA34 filters counter-trend trades; volume confirmation reduces false breakouts.
Designed for low trade frequency (target: 20-40/year) to minimize fee drag and work in both bull and bear markets via trend filter.
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
    
    # Calculate 4h Camarilla levels (based on previous day's OHLC)
    # For 4h data, we approximate using 16-period lookback (4h * 16 = ~4 days ~ 1 day)
    lookback = 16
    high_lookback = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    low_lookback = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    close_lookback = pd.Series(close).rolling(window=lookback, min_periods=lookback).last().values
    
    # Camarilla levels calculation
    range_val = high_lookback - low_lookback
    camarilla_r1 = close_lookback + range_val * 1.1 / 12
    camarilla_s1 = close_lookback - range_val * 1.1 / 12
    
    # Calculate 12h EMA34 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Calculate 4h volume MA (20-period) for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(lookback, 34, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(high_lookback[i]) or np.isnan(low_lookback[i]) or np.isnan(close_lookback[i]) or
            np.isnan(camarilla_r1[i]) or np.isnan(camarilla_s1[i]) or
            np.isnan(ema_34_12h_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: 12h close > EMA34 = uptrend, close < EMA34 = downtrend
        close_12h_aligned = align_htf_to_ltf(prices, df_12h, close_12h)
        trend_up = close_12h_aligned[i] > ema_34_12h_aligned[i]
        trend_down = close_12h_aligned[i] < ema_34_12h_aligned[i]
        
        # Volume filter: 4h volume > 1.8x 20-period MA
        vol_filter = volume[i] > 1.8 * vol_ma_20[i]
        
        if position == 0:
            # Long: price breaks above Camarilla R1 AND uptrend AND volume filter
            if close[i] > camarilla_r1[i] and trend_up and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Camarilla S1 AND downtrend AND volume filter
            elif close[i] < camarilla_s1[i] and trend_down and vol_filter:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: price retouches Camarilla S1 (mean reversion) OR 12h trend turns down
                if close[i] < camarilla_s1[i] or not trend_up:
                    exit_signal = True
            elif position == -1:
                # Short exit: price retouches Camarilla R1 (mean reversion) OR 12h trend turns up
                if close[i] > camarilla_r1[i] or not trend_down:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Camarilla_R1S1_Breakout_12hEMA34_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0