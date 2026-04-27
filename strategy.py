#!/usr/bin/env python3
"""
1h_Camarilla_R3_S3_Breakout_4hTrend_VolumeConfirm
Hypothesis: On 1h timeframe, Camarilla R3/S3 breakouts aligned with 4h EMA50 trend and volume confirmation capture sustained moves while minimizing trades. Using 4h for trend direction reduces whipsaws, and 1h only for precise entry timing keeps trade frequency in the 15-35/year range. Discrete sizing (0.20) limits fee churn. Works in both bull (trend-following breaks) and bear (mean-reversion at extremes) via trend filter.
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
    
    # Get 4h data for EMA50 trend filter (HTF direction)
    df_4h = get_htf_data(prices, '4h')
    
    # Get 1d data for Camarilla levels (structure)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h EMA50 for trend filter
    close_4h_series = pd.Series(df_4h['close'].values)
    ema_50_4h = close_4h_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate Camarilla levels from previous 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla R3, S3 levels: R3 = close + 1.1*(high-low)/2, S3 = close - 1.1*(high-low)/2
    rng_1d = high_1d - low_1d
    camarilla_r3 = close_1d + 1.1 * rng_1d / 2
    camarilla_s3 = close_1d - 1.1 * rng_1d / 2
    
    # Align all indicators to primary timeframe (1h)
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume confirmation: current volume > 1.5 * 24-period average (1d equivalent)
    vol_avg = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_confirm = volume > (1.5 * vol_avg)
    
    # Session filter: 08-20 UTC (reduce noise outside active hours)
    hours = prices.index.hour  # open_time is already datetime64[ms]
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.20   # Position size: 20% of capital (discrete level)
    
    # Warmup: need 4h EMA50 (50), volume avg (24)
    start_idx = max(50, 24)
    
    for i in range(start_idx, n):
        # Skip if any data not ready or outside session
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(volume_confirm[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        ema_4h_val = ema_50_4h_aligned[i]
        r3_val = r3_aligned[i]
        s3_val = s3_aligned[i]
        vol_conf = volume_confirm[i]
        
        if position == 0:
            # Determine trend: price relative to 4h EMA50
            is_uptrend = close_val > ema_4h_val
            is_downtrend = close_val < ema_4h_val
            
            if is_uptrend:
                # Uptrend: long when price breaks above R3 and volume confirms
                if (close_val > r3_val) and vol_conf:
                    signals[i] = size
                    position = 1
            elif is_downtrend:
                # Downtrend: short when price breaks below S3 and volume confirms
                if (close_val < s3_val) and vol_conf:
                    signals[i] = -size
                    position = -1
        elif position == 1:
            # Exit long: price touches S3 (support) or trend changes to downtrend
            exit_condition = (close_val < s3_val) or (close_val < ema_4h_val)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price touches R3 (resistance) or trend changes to uptrend
            exit_condition = (close_val > r3_val) or (close_val > ema_4h_val)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1h_Camarilla_R3_S3_Breakout_4hTrend_VolumeConfirm"
timeframe = "1h"
leverage = 1.0