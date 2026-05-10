# #!/usr/bin/env python3
# 12H_Camarilla_R3S3_Breakout_1wTrend_VolumeSpike_Dyn
# Hypothesis: Breakouts at weekly Camarilla R3/S3 levels on 12h timeframe with volume confirmation (>1.5x 20-period average) and 1-week trend alignment (price > EMA 50 for long, < EMA 50 for short). Designed for low trade frequency (~25-35/year) to minimize fee drag while capturing strong momentum moves in both bull and bear markets. Uses discrete position sizing (0.25) and dynamic exit at opposite Camarilla level (S1 for longs, R1 for shorts) to lock in gains.

name = "12H_Camarilla_R3S3_Breakout_1wTrend_VolumeSpike_Dyn"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1-week Camarilla pivot levels (from previous week)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 1w bar
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    range_1w = high_1w - low_1w
    # S3 = close - (range * 1.2500)
    # R3 = close + (range * 1.2500)
    s3 = close_1w - (range_1w * 1.25000)
    r3 = close_1w + (range_1w * 1.25000)
    
    # Align to 12h timeframe (wait for 1w bar to close)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    
    # 1-week trend filter: EMA 50
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_threshold = vol_ma * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough history for indicators
    
    for i in range(start_idx, n):
        if np.isnan(s3_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_threshold[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine 1w trend
        is_uptrend = close[i] > ema_50_1w_aligned[i]
        is_downtrend = close[i] < ema_50_1w_aligned[i]
        
        if position == 0:
            # Long entry: Price breaks above R3 + volume confirmation + 1w uptrend
            if (close[i] > r3_aligned[i] and 
                volume[i] > vol_threshold[i] and 
                is_uptrend):
                signals[i] = 0.25
                position = 1
            # Short entry: Price breaks below S3 + volume confirmation + 1w downtrend
            elif (close[i] < s3_aligned[i] and 
                  volume[i] > vol_threshold[i] and 
                  is_downtrend):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Price crosses below S3 (opposite side) OR reverse signal
            if close[i] < s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Price crosses above R3 (opposite side) OR reverse signal
            if close[i] > r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals