#!/usr/bin/env python3
"""
12h_Camarilla_R3_S3_Breakout_1dTrend_VolumeConfirm
Hypothesis: Camarilla R3/S3 levels from 12h chart act as strong support/resistance. 
Breakout above R3 with volume confirmation and 1d uptrend (price > EMA34) goes long.
Breakdown below S3 with volume confirmation and 1d downtrend (price < EMA34) goes short.
Exits on opposite Camarilla level touch or trend reversal. Uses discrete sizing (0.25) 
to limit fee churn. Designed for 12h timeframe targeting 50-150 total trades over 4 years 
(12-37/year). Works in bull markets via upside breakouts and bear markets via 
downside breakdowns, with volume filter preventing false breakouts in low-liquidity periods.
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
    
    # Get 12h data for Camarilla levels and 1d data for trend filter
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 12h Camarilla levels: R3, S3
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    rang = high_12h - low_12h
    R3 = close_12h + rang * 1.1 / 4
    S3 = close_12h - rang * 1.1 / 4
    
    # Calculate 1d EMA34 for trend filter
    close_1d_series = pd.Series(df_1d['close'].values)
    ema_34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align all indicators to primary timeframe (12h)
    R3_aligned = align_htf_to_ltf(prices, df_12h, R3)
    S3_aligned = align_htf_to_ltf(prices, df_12h, S3)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: current volume > 2.0 * 20-period average (tighter to reduce trades)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital (discrete level)
    
    # Warmup: need EMA34 (34), volume avg (20)
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        R3_val = R3_aligned[i]
        S3_val = S3_aligned[i]
        ema_1d_val = ema_34_1d_aligned[i]
        vol_conf = volume_confirm[i]
        
        if position == 0:
            # Determine 1d trend: price > EMA34 = uptrend, price < EMA34 = downtrend
            is_uptrend = close_val > ema_1d_val
            is_downtrend = close_val < ema_1d_val
            
            if is_uptrend:
                # Uptrend: long when price breaks above R3 and volume confirms
                if (close_val > R3_val) and vol_conf:
                    signals[i] = size
                    position = 1
            elif is_downtrend:
                # Downtrend: short when price breaks below S3 and volume confirms
                if (close_val < S3_val) and vol_conf:
                    signals[i] = -size
                    position = -1
        elif position == 1:
            # Exit long: price touches S3 (support) or trend changes to downtrend
            exit_condition = (close_val < S3_val) or (close_val < ema_1d_val)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price touches R3 (resistance) or trend changes to uptrend
            exit_condition = (close_val > R3_val) or (close_val > ema_1d_val)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_Camarilla_R3_S3_Breakout_1dTrend_VolumeConfirm"
timeframe = "12h"
leverage = 1.0