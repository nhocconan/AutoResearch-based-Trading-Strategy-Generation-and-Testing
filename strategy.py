#!/usr/bin/env python3
"""
12h_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike_v1
Hypothesis: Camarilla R3/S3 breakouts with 1d EMA34 trend filter and volume spikes capture strong directional moves in both bull and bear markets. The 12h timeframe minimizes fee drag while allowing sufficient trades. Discrete sizing (0.25) controls drawdown. Uses proven Camarilla pivot structure with volume confirmation and trend alignment.
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
    
    # Load 1d data ONCE before loop for Camarilla pivots and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Camarilla pivots from previous 1d bar
    # Pivot = (H + L + C) / 3
    # R3 = C + (H - L) * 1.1/2
    # S3 = C - (H - L) * 1.1/2
    prev_close = df_1d['close'].shift(1)
    prev_high = df_1d['high'].shift(1)
    prev_low = df_1d['low'].shift(1)
    
    pivot = (prev_high + prev_low + prev_close) / 3
    r3 = prev_close + (prev_high - prev_low) * 1.1 / 2
    s3 = prev_close - (prev_high - prev_low) * 1.1 / 2
    
    # 1d EMA34 for trend filter
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align to 12h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3.values)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3.values)
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Warmup: max of EMA34 (34), volume MA (20)
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        close_val = close[i]
        r3_val = r3_aligned[i]
        s3_val = s3_aligned[i]
        ema_val = ema34_aligned[i]
        vol_conf = volume_confirm[i]
        
        # Skip if any data not ready
        if (np.isnan(r3_val) or np.isnan(s3_val) or np.isnan(ema_val)):
            # Hold current position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        # Trend filter: price above/below EMA34
        is_uptrend = close_val > ema_val
        is_downtrend = close_val < ema_val
        
        # Breakout conditions with volume confirmation
        long_breakout = close_val > r3_val and vol_conf
        short_breakout = close_val < s3_val and vol_conf
        
        # Exit conditions: reverse breakout or loss of momentum
        long_exit = (position == 1 and close_val < s3_val)
        short_exit = (position == -1 and close_val > r3_val)
        
        if long_breakout and is_uptrend and position != 1:
            signals[i] = base_size
            position = 1
        elif short_breakout and is_downtrend and position != -1:
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

name = "12h_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0