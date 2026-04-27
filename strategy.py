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
    
    # Get weekly data for trend filter and pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Previous week's high, low, close
    prev_high = df_1w['high'].shift(1).values
    prev_low = df_1w['low'].shift(1).values
    prev_close = df_1w['close'].shift(1).values
    
    # Calculate Camarilla levels
    range_hl = prev_high - prev_low
    R3 = prev_close + range_hl * 1.1 / 4
    S3 = prev_close - range_hl * 1.1 / 4
    
    # Align Camarilla levels to daily timeframe
    R3_1d = align_htf_to_ltf(prices, df_1w, R3)
    S3_1d = align_htf_to_ltf(prices, df_1w, S3)
    
    # Get weekly EMA34 for trend filter
    close_1w = df_1w['close'].values
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Volume spike: current volume > 2.0 * 5-period average (5-day lookback)
    vol_ma_5 = np.full(n, np.nan)
    for i in range(5, n):
        vol_ma_5[i] = np.mean(volume[i-5:i])
    volume_spike = volume > (2.0 * vol_ma_5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need enough data for calculations
    start_idx = max(34, 5) + 1
    
    for i in range(start_idx, n):
        if (np.isnan(R3_1d[i]) or np.isnan(S3_1d[i]) or 
            np.isnan(ema34_1w_aligned[i]) or np.isnan(vol_ma_5[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long entry: price breaks above R3 + 1-week uptrend + volume spike
            if (close[i] > R3_1d[i] and close[i] > ema34_1w_aligned[i] and volume_spike[i]):
                signals[i] = 0.30
                position = 1
            # Short entry: price breaks below S3 + 1-week downtrend + volume spike
            elif (close[i] < S3_1d[i] and close[i] < ema34_1w_aligned[i] and volume_spike[i]):
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price breaks below S3 (reversal) or trend changes
            if (close[i] < S3_1d[i] or close[i] < ema34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Short exit: price breaks above R3 (reversal) or trend changes
            if (close[i] > R3_1d[i] or close[i] > ema34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

name = "1d_Camarilla_R3S3_Breakout_1wTrend_VolumeSpike"
timeframe = "1d"
leverage = 1.0