#!/usr/bin/env python3
# 4h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike
# Hypothesis: Camarilla pivot levels from daily timeframe act as key support/resistance zones.
# Price breaking above R3 or below S3 with volume spike and aligned daily trend indicates
# strong momentum continuation. Works in bull via R3 breaks, bear via S3 breaks.
# Volume filter reduces false breakouts, trend filter ensures directional alignment.
# Target: 20-50 trades per year (~80-200 over 4 years) with position size 0.25.

name = "4h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE for Camarilla levels and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    # R3 = close + (high - low) * 1.1/2
    # S3 = close - (high - low) * 1.1/2
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    range_ = prev_high - prev_low
    R3 = prev_close + range_ * 1.1 / 2
    S3 = prev_close - range_ * 1.1 / 2
    
    # Align Camarilla levels to 4h timeframe (already delayed by shift(1))
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    
    # Daily EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume ratio: current volume / 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.where(vol_ma > 0, volume / vol_ma, 1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need volume MA
    
    for i in range(start_idx, n):
        if np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or np.isnan(ema_34_aligned[i]) or np.isnan(vol_ratio[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Breakout conditions: price breaks above R3 or below S3
        breakout_up = close[i] > R3_aligned[i]
        breakout_down = close[i] < S3_aligned[i]
        
        # Volume confirmation: volume > 2x average
        volume_confirm = vol_ratio[i] > 2.0
        
        # Trend filter from daily EMA34
        uptrend = close[i] > ema_34_aligned[i]
        downtrend = close[i] < ema_34_aligned[i]
        
        if position == 0:
            # Long: upward breakout + volume + uptrend
            if breakout_up and volume_confirm and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: downward breakout + volume + downtrend
            elif breakout_down and volume_confirm and downtrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price breaks back below S3 or trend reversal
            if close[i] < S3_aligned[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price breaks back above R3 or trend reversal
            if close[i] > R3_aligned[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals