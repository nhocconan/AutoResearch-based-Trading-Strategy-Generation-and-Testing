#!/usr/bin/env python3
"""
12h_Camarilla_R3_S3_Breakout_1dTrend_VolumeConfirm
Hypothesis: Camarilla R3/S3 breakout on 12h with 1d EMA34 trend filter and volume spike confirmation.
Long when price breaks above R3 with 1d bullish trend and volume > 1.5x average.
Short when price breaks below S3 with 1d bearish trend and volume > 1.5x average.
Uses discrete sizing (0.25) to minimize fee drag. Target: 50-150 trades over 4 years.
Works in bull/bear via 1d EMA trend filter and volume confirmation to avoid false breakouts.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla pivot levels for 12h
    # Pivot = (high + low + close) / 3
    # R3 = close + (high - low) * 1.1 / 4
    # S3 = close - (high - low) * 1.1 / 4
    pivot = (high + low + close) / 3
    r3 = close + (high - low) * 1.1 / 4
    s3 = close - (high - low) * 1.1 / 4
    
    # Load 1d data for HTF trend filter and volume average
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 1d average volume for confirmation (20-period)
    avg_vol_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    avg_vol_20_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_vol_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Start after warmup (need 34-period for EMA and 20 for volume avg)
    start_idx = 34
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r3[i]) or np.isnan(s3[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(avg_vol_20_1d_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Volume confirmation: current volume > 1.5x 1d average volume
        volume_confirmed = volume[i] > 1.5 * avg_vol_20_1d_aligned[i]
        
        # Trend filter: price vs 1d EMA34
        trend_bullish = close[i] > ema_34_1d_aligned[i]
        trend_bearish = close[i] < ema_34_1d_aligned[i]
        
        # Breakout conditions
        breakout_long = close[i] > r3[i]
        breakout_short = close[i] < s3[i]
        
        # Long logic: breakout above R3 + bullish trend + volume confirmation
        if breakout_long and trend_bullish and volume_confirmed:
            if position != 1:
                signals[i] = base_size
                position = 1
            else:
                signals[i] = base_size
        # Short logic: breakout below S3 + bearish trend + volume confirmation
        elif breakout_short and trend_bearish and volume_confirmed:
            if position != -1:
                signals[i] = -base_size
                position = -1
            else:
                signals[i] = -base_size
        # Exit conditions: reverse breakout or trend change
        elif position == 1 and (close[i] < s3[i] or close[i] < ema_34_1d_aligned[i]):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (close[i] > r3[i] or close[i] > ema_34_1d_aligned[i]):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
    
    return signals

name = "12h_Camarilla_R3_S3_Breakout_1dTrend_VolumeConfirm"
timeframe = "12h"
leverage = 1.0