#!/usr/bin/env python3
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
    
    # Get weekly data for trend filter (1w HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Get daily data for pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous week's high, low, close (1w shift for completed week)
    prev_high_w = df_1w['high'].shift(1).values
    prev_low_w = df_1w['low'].shift(1).values
    prev_close_w = df_1w['close'].shift(1).values
    
    # Calculate Weekly Pivot Levels (R1 and S1)
    range_hl_w = prev_high_w - prev_low_w
    PP_w = (prev_high_w + prev_low_w + prev_close_w) / 3
    R1_w = 2 * PP_w - prev_low_w
    S1_w = 2 * PP_w - prev_high_w
    
    # Align Weekly Pivot Levels to daily timeframe
    R1_w_daily = align_htf_to_ltf(prices, df_1w, R1_w)
    S1_w_daily = align_htf_to_ltf(prices, df_1w, S1_w)
    
    # Get weekly EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume spike: current volume > 2.0 * 20-period average (20-day lookback)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need enough data for calculations
    start_idx = max(50, 20) + 1
    
    for i in range(start_idx, n):
        if (np.isnan(R1_w_daily[i]) or np.isnan(S1_w_daily[i]) or 
            np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long entry: price breaks above R1_w + 1-week uptrend + volume spike
            if (close[i] > R1_w_daily[i] and close[i] > ema50_1w_aligned[i] and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below S1_w + 1-week downtrend + volume spike
            elif (close[i] < S1_w_daily[i] and close[i] < ema50_1w_aligned[i] and volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price breaks below S1_w (reversal) or trend changes
            if (close[i] < S1_w_daily[i] or close[i] < ema50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above R1_w (reversal) or trend changes
            if (close[i] > R1_w_daily[i] or close[i] > ema50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WeeklyPivot_R1S1_Breakout_1wTrend_VolumeSpike"
timeframe = "1d"
leverage = 1.0