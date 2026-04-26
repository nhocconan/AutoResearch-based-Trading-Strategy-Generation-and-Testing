#!/usr/bin/env python3
"""
6h_Camarilla_R3_S3_Breakout_1dEMA34_Trend_VolumeSpike_v1
Hypothesis: On 6h timeframe, price breaking Camarilla R3/S3 levels with 1d EMA34 trend alignment and volume confirmation provides robust breakout signals. The 1d EMA34 offers a balanced trend filter that works in both bull and bear markets by capturing the primary trend while reducing noise. Volume confirmation (2.0x average) ensures breakouts have conviction. Discrete sizing (0.0, ±0.25) minimizes fee churn. Targets ~12-30 trades/year (~50-120 over 4 years) to stay within optimal trade frequency for 6h timeframe.
"""

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
    
    # Get 1d data for HTF trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate EMA(34) on 1d for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate volume ratio (current / 20-period average) for spike confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.maximum(vol_ma, 1e-10)  # avoid division by zero
    
    # Calculate Camarilla levels from previous 6h bar
    prev_high = np.concatenate([[np.nan], high[:-1]])
    prev_low = np.concatenate([[np.nan], low[:-1]])
    prev_close = np.concatenate([[np.nan], close[:-1]])
    
    camarilla_r3 = prev_close + ((prev_high - prev_low) * 1.1 / 4)
    camarilla_s3 = prev_close - ((prev_high - prev_low) * 1.1 / 4)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: max of 1d EMA(34), volume MA(20)
    start_idx = max(34, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(vol_ratio[i]) or
            np.isnan(camarilla_r3[i]) or
            np.isnan(camarilla_s3[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        close_val = close[i]
        vol_confirmed = vol_ratio[i] > 2.0  # volume at least 2.0x average (stricter)
        trend_1d_up = close_val > ema_34_1d_aligned[i]
        trend_1d_down = close_val < ema_34_1d_aligned[i]
        
        if position == 0:
            # Long: price breaks above Camarilla R3 AND 1d trend up AND volume confirmation
            long_signal = (close_val > camarilla_r3[i]) and trend_1d_up and vol_confirmed
            
            # Short: price breaks below Camarilla S3 AND 1d trend down AND volume confirmation
            short_signal = (close_val < camarilla_s3[i]) and trend_1d_down and vol_confirmed
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: trend flips down
            if not trend_1d_up:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: trend flips up
            if not trend_1d_down:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Camarilla_R3_S3_Breakout_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0