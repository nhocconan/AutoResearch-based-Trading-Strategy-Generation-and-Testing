#!/usr/bin/env python3
"""
12h_Camarilla_R3_S3_Breakout_1wTrend_VolumeSpike
Hypothesis: On 12h timeframe, Camarilla R3/S3 breakouts aligned with 1w trend and volume confirmation capture sustained moves in both bull and bear markets. 
In bull markets, upside breakouts above R3 with 1w uptrend go long; in bear markets, downside breakdowns below S3 with 1w downtrend go short. 
Volume filter (>2.0x 24-bar average) reduces false breakouts. Exits on opposite Camarilla level (R3/S3) or 1w trend reversal. 
Discrete sizing (0.25) limits fee churn. Target: 50-150 total trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 80:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for Camarilla pivot levels and trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate Camarilla levels from previous 1w bar
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Camarilla R3, S3 levels: based on previous week's range
    # R3 = close + 1.1 * (high - low) / 2
    # S3 = close - 1.1 * (high - low) / 2
    rng = high_1w - low_1w
    camarilla_r3 = close_1w + 1.1 * rng / 2
    camarilla_s3 = close_1w - 1.1 * rng / 2
    
    # Calculate 1w EMA34 for trend filter
    close_1w_series = pd.Series(close_1w)
    ema_34_1w = close_1w_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align all indicators to primary timeframe (12h)
    r3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s3)
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume confirmation: current volume > 2.0 * 24-period average (12h equivalent: 24*12h = 12d)
    vol_avg = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_confirm = volume > (2.0 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital (discrete level)
    
    # Warmup: need 1w EMA34 (34) and volume avg (24)
    start_idx = max(34, 24)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        r3_val = r3_aligned[i]
        s3_val = s3_aligned[i]
        ema_1w_val = ema_34_1w_aligned[i]
        vol_conf = volume_confirm[i]
        
        if position == 0:
            # Determine 1w trend: price > EMA34 = uptrend, price < EMA34 = downtrend
            is_uptrend = close_val > ema_1w_val
            is_downtrend = close_val < ema_1w_val
            
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
            # Exit long: price touches S3 (support) or 1w trend changes to downtrend
            exit_condition = (close_val < s3_val) or (close_val < ema_1w_val)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price touches R3 (resistance) or 1w trend changes to uptrend
            exit_condition = (close_val > r3_val) or (close_val > ema_1w_val)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_Camarilla_R3_S3_Breakout_1wTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0