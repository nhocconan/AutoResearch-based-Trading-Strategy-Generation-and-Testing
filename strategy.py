#!/usr/bin/env python3
"""
4h_Camarilla_R3S3_Breakout_1dTrend_VolumeConfirm_v1
Hypothesis: Trade Camarilla R3/S3 breakouts on 4h with 1d trend filter (price > EMA200) and volume spike confirmation.
Long when price breaks above R3 in uptrend with volume > 1.5x 20-period average.
Short when price breaks below S3 in downtrend with volume > 1.5x 20-period average.
Exit on opposite level touch (R3/S3) or trend reversal.
Position size: 0.25 to limit drawdown and enable multiple concurrent positions across symbols.
Target: 20-40 trades/year to stay well under 400-trade 4h hard max.
Works in bull markets (breakouts with uptrend) and bear markets (breakdowns with downtrend).
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
    
    # Get 1d data for HTF trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need sufficient data for EMA200
        return np.zeros(n)
    
    # Calculate 1d EMA200 for HTF trend filter
    close_1d = df_1d['close'].values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Calculate Camarilla levels from previous 1d bar
    # Camarilla: based on previous day's range
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_prev = df_1d['close'].shift(1).values  # Previous day close
    
    # Calculate Camarilla levels for each 1d bar
    camarilla_R3 = np.full_like(close_1d, np.nan)
    camarilla_S3 = np.full_like(close_1d, np.nan)
    
    # Camarilla formulas:
    # R3 = close + 1.1 * (high - low) / 4
    # S3 = close - 1.1 * (high - low) / 4
    for i in range(len(df_1d)):
        if i == 0:  # First bar has no previous day
            continue
        daily_range = high_1d[i-1] - low_1d[i-1]
        camarilla_R3[i] = close_1d[i-1] + 1.1 * daily_range / 4
        camarilla_S3[i] = close_1d[i-1] - 1.1 * daily_range / 4
    
    # Align Camarilla levels to 4h timeframe
    camarilla_R3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R3)
    camarilla_S3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S3)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA200 (200) and volume MA (20)
    start_idx = max(200, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_200_1d_aligned[i]) or 
            np.isnan(camarilla_R3_aligned[i]) or np.isnan(camarilla_S3_aligned[i]) or
            np.isnan(volume_ma[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine 1d HTF trend (bullish = price above EMA200)
        htf_1d_bullish = close[i] > ema_200_1d_aligned[i]
        htf_1d_bearish = close[i] < ema_200_1d_aligned[i]
        
        if position == 0:
            # Long setup: price breaks above R3 + 1d uptrend + volume confirmation
            long_setup = (close[i] > camarilla_R3_aligned[i]) and htf_1d_bullish and volume_confirm[i]
            
            # Short setup: price breaks below S3 + 1d downtrend + volume confirmation
            short_setup = (close[i] < camarilla_S3_aligned[i]) and htf_1d_bearish and volume_confirm[i]
            
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
            # Exit: price touches S3 (stop) OR 1d trend turns bearish
            if (close[i] <= camarilla_S3_aligned[i]) or (not htf_1d_bullish):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price touches R3 (stop) OR 1d trend turns bullish
            if (close[i] >= camarilla_R3_aligned[i]) or (htf_1d_bullish):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R3S3_Breakout_1dTrend_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0