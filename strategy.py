#!/usr/bin/env python3
"""
4h_Camarilla_R3S3_Breakout_12hTrend_VolumeSpike_v2
Hypothesis: Camarilla R3/S3 breakout with 12h EMA50 trend filter and volume spike captures strong momentum moves while avoiding false breakouts in chop. Uses 12h trend for better alignment with 4h signals, targeting 20-50 trades/year with discrete sizing (0.25). Designed to work in both bull and bear markets by requiring trend alignment.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # 12h EMA50 for trend filter
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Previous 12h bar's Camarilla levels (using 12h OHLC)
    prev_close = df_12h['close'].shift(1).values
    prev_high = df_12h['high'].shift(1).values
    prev_low = df_12h['low'].shift(1).values
    
    # Camarilla calculations
    R3 = prev_close + (prev_high - prev_low) * 1.1 / 4
    S3 = prev_close - (prev_high - prev_low) * 1.1 / 4
    R4 = prev_close + (prev_high - prev_low) * 1.1 / 2
    S4 = prev_close - (prev_high - prev_low) * 1.1 / 2
    
    # Align to 4h timeframe
    R3_aligned = align_htf_to_ltf(prices, df_12h, R3)
    S3_aligned = align_htf_to_ltf(prices, df_12h, S3)
    R4_aligned = align_htf_to_ltf(prices, df_12h, R4)
    S4_aligned = align_htf_to_ltf(prices, df_12h, S4)
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Warmup: max of EMA50 (50) and volume MA (20)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        close_val = close[i]
        ema_val = ema_50_12h_aligned[i]
        r3_val = R3_aligned[i]
        s3_val = S3_aligned[i]
        r4_val = R4_aligned[i]
        s4_val = S4_aligned[i]
        vol_spike = volume_spike[i]
        
        # Skip if any data not ready
        if (np.isnan(ema_val) or np.isnan(r3_val) or np.isnan(s3_val) or 
            np.isnan(r4_val) or np.isnan(s4_val)):
            # Hold current position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        # Trend filter: price vs 12h EMA50
        uptrend = close_val > ema_val
        downtrend = close_val < ema_val
        
        # Long: price breaks above R3 with 12h uptrend and volume spike
        long_condition = (close_val > r3_val) and uptrend and vol_spike
        # Short: price breaks below S3 with 12h downtrend and volume spike
        short_condition = (close_val < s3_val) and downtrend and vol_spike
        
        # Exit: price re-enters R3-S3 range
        long_exit = (position == 1 and close_val < r3_val)
        short_exit = (position == -1 and close_val > s3_val)
        
        if long_condition and position != 1:
            signals[i] = base_size
            position = 1
        elif short_condition and position != -1:
            signals[i] = -base_size
            position = -1
        elif long_exit:
            signals[i] = 0.0
            position = 0
        elif short_exit:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
    
    return signals

name = "4h_Camarilla_R3S3_Breakout_12hTrend_VolumeSpike_v2"
timeframe = "4h"
leverage = 1.0